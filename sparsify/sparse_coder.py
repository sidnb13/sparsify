import json
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal, NamedTuple

import einops
import torch
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors import safe_open
from safetensors.torch import load_model, save_model
from torch import Tensor, nn

from .config import PKMConfig, SparseCoderConfig
from .fused_encoder import EncoderOutput, fused_encoder
from .pkm import PKM
from .utils import decoder_impl


class ForwardOutput(NamedTuple):
    sae_out: Tensor

    latent_acts: Tensor
    """Activations of the top-k latents."""

    latent_indices: Tensor
    """Indices of the top-k features."""

    fvu: Tensor
    """Fraction of variance unexplained."""

    auxk_loss: Tensor
    """AuxK loss, if applicable."""

    multi_topk_fvu: Tensor
    """Multi-TopK FVU, if applicable."""
    
    encoding_flops: int
    """FLOPs for the encoding step."""
    
    decoding_flops: int
    """FLOPs for the decoding step."""
    
    total_flops: int
    """Total FLOPs for the forward pass."""


class SparseCoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
        pkm_cfg: PKMConfig | None = None,
        encode_method: Literal["linear", "pkm"] = "linear",
        ctx_len: int = 128,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor
        self.encode_method = encode_method

        if encode_method == "linear":
            self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
            self.encoder.bias.data.zero_()
        elif encode_method == "pkm":
            assert pkm_cfg is not None, "PKM configuration is required."
            self.pkm = PKM(
                d_in,
                self.num_latents,
                ctx_len=ctx_len,
                topk=cfg.k,
                input_dropout=pkm_cfg.input_dropout,
                query_dropout=pkm_cfg.query_dropout,
                pre_layernorm=pkm_cfg.pre_layernorm,
            )
            self.pkm = self.pkm.to(device)

        if decoder:
            if encode_method == "linear":
                # Transcoder initialization: use zeros
                if cfg.transcode:
                    self.W_dec = nn.Parameter(
                        torch.zeros_like(self.encoder.weight.data)
                    )
                # Sparse autoencoder initialization: 
                # use the transpose of encoder weights
                else:
                    self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
                    if self.cfg.normalize_decoder:
                        self.set_decoder_norm_to_unit_norm()
            elif encode_method == "pkm":
                # For PKM, initialize decoder weights randomly
                self.W_dec = nn.Parameter(
                    torch.randn(self.num_latents, d_in, device=device, dtype=dtype)
                )
                if self.cfg.normalize_decoder:
                    self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))
        self.W_skip = (
            nn.Parameter(torch.zeros(d_in, d_in, device=device, dtype=dtype))
            if cfg.skip_connection
            else None
        )

    @staticmethod
    def load_many(
        name: str,
        local: bool = False,
        layers: list[str] | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
        pattern: str | None = None,
    ) -> dict[str, "SparseCoder"]:
        """Load sparse coders for multiple hookpoints on a single model and dataset."""
        pattern = pattern + "/*" if pattern is not None else None
        if local:
            repo_path = Path(name)
        else:
            repo_path = Path(snapshot_download(name, allow_patterns=pattern))

        if layers is not None:
            return {
                layer: SparseCoder.load_from_disk(
                    repo_path / layer, device=device, decoder=decoder
                )
                for layer in natsorted(layers)
            }
        files = [
            f
            for f in repo_path.iterdir()
            if f.is_dir() and (pattern is None or fnmatch(f.name, pattern))
        ]
        return {
            f.name: SparseCoder.load_from_disk(f, device=device, decoder=decoder)
            for f in natsorted(files, key=lambda f: f.name)
        }

    @staticmethod
    def load_from_hub(
        name: str,
        hookpoint: str | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "SparseCoder":
        # Download from the HuggingFace Hub
        repo_path = Path(
            snapshot_download(
                name,
                allow_patterns=f"{hookpoint}/*" if hookpoint is not None else None,
            )
        )
        if hookpoint is not None:
            repo_path = repo_path / hookpoint

        # No layer specified, and there are multiple layers
        elif not repo_path.joinpath("cfg.json").exists():
            raise FileNotFoundError("No config file found; try specifying a layer.")

        return SparseCoder.load_from_disk(repo_path, device=device, decoder=decoder)

    @staticmethod
    def load_from_disk(
        path: Path | str,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "SparseCoder":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = SparseCoderConfig.from_dict(cfg_dict, drop_extra_fields=True)

        safetensors_path = str(path / "sae.safetensors")

        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            first_key = next(iter(f.keys()))
            reference_dtype = f.get_tensor(first_key).dtype

        sae = SparseCoder(
            d_in, cfg, device=device, decoder=decoder, dtype=reference_dtype
        )

        load_model(
            model=sae,
            filename=safetensors_path,
            device=str(device),
            # TODO: Maybe be more fine-grained about this in the future?
            strict=decoder,
        )
        return sae

    def save_to_disk(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump(
                {
                    **self.cfg.to_dict(),
                    "d_in": self.d_in,
                },
                f,
            )

    @property
    def device(self):
        if self.encode_method == "linear":
            return self.encoder.weight.device
        else:  # PKM
            return next(self.pkm.parameters()).device

    @property
    def dtype(self):
        if self.encode_method == "linear":
            return self.encoder.weight.dtype
        else:  # PKM
            return next(self.pkm.parameters()).dtype

    def _manual_encode_flops(self, x: Tensor) -> int:
        """Manually count encoding FLOPs based on the encoding method."""
        batch_size = x.shape[0]
        
        if self.encode_method == "linear":
            # Full linear layer: batch * d_in * num_latents
            # This includes the matrix multiply
            linear_flops = batch_size * self.d_in * self.num_latents
            # ReLU: batch * num_latents (element-wise)
            relu_flops = batch_size * self.num_latents
            # TopK: approximate as batch * num_latents comparisons
            topk_flops = batch_size * self.num_latents
            return linear_flops + relu_flops + topk_flops
            
        elif self.encode_method == "pkm":
            import math
            # PKM uses factorized product keys
            # Split into 2 subspaces, sqrt(n) keys per subspace
            num_keys = int(math.sqrt(self.num_latents))
            half_dim = self.d_in // 2
            
            # Query projection and norm (2 heads)
            # LayerNorm: ~4 * batch * d_in (mean, var, scale, shift)
            ln_flops = 4 * batch_size * self.d_in
            
            # Einsum: "p b t d, n p d -> b t p n"
            # For each of 2 heads (p=2): batch * num_keys * half_dim
            einsum_flops = 2 * batch_size * num_keys * half_dim
            
            # TopK per head: 2 * batch * num_keys
            topk_per_head = 2 * batch_size * num_keys
            
            # Cartesian product and final topK: batch * topk^2
            cartesian_flops = batch_size * self.cfg.k * self.cfg.k
            
            return ln_flops + einsum_flops + topk_per_head + cartesian_flops
        
        return 0
    
    def _manual_decode_flops(self, top_acts: Tensor) -> int:
        """Manually count decoding FLOPs."""
        batch_size = top_acts.shape[0]
        k = top_acts.shape[1]
        # Sparse matmul: batch * k * d_in
        return batch_size * k * self.d_in

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        if not self.cfg.transcode:
            x = x - self.b_dec

        if self.encode_method == "linear":
            return fused_encoder(
                x,
                self.encoder.weight,
                self.encoder.bias,
                self.cfg.k,
                self.cfg.activation,
            )
        elif self.encode_method == "pkm":
            return self.pkm(x)
        else:
            raise ValueError(f"Unknown encode method: {self.encode_method}")

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."

        y = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        return y + self.b_dec

    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        # Use autocast only on CUDA with bf16 support (not on MPS)
        device_type = x.device.type
        use_autocast = device_type == "cuda" and torch.cuda.is_bf16_supported()
        
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_autocast,
        ):
            return self._forward_impl(x, y, dead_mask=dead_mask)

    def _forward_impl(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        # Count encoding FLOPs
        encode_flops = self._manual_encode_flops(x)
        top_acts, top_indices, pre_acts = self.encode(x)

        # If we aren't given a distinct target, we're autoencoding
        if y is None:
            y = x

        # Count decoding FLOPs
        decode_flops = self._manual_decode_flops(top_acts)
        sae_out = self.decode(top_acts, top_indices)
        if self.W_skip is not None:
            sae_out += x.to(self.dtype) @ self.W_skip.mT

        # Compute the residual
        e = y - sae_out

        # Used as a denominator for putting everything on a reasonable scale
        total_variance = (y - y.mean(0)).pow(2).sum()
        # Add small epsilon to prevent division by zero
        total_variance = torch.clamp(total_variance, min=1e-8)

        # Second decoder pass for AuxK loss
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # Heuristic from Appendix B.1 in the paper
            k_aux = y.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Don't include living latents in this loss
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            # Encourage the top ~50% of dead latents to predict the residual of the
            # top k living latents
            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        if self.cfg.multi_topk:
            top_acts, top_indices = pre_acts.topk(4 * self.cfg.k, sorted=False)
            sae_out = self.decode(top_acts, top_indices)

            multi_topk_fvu = (sae_out - y).pow(2).sum() / total_variance
        else:
            multi_topk_fvu = sae_out.new_tensor(0.0)

        # Get FLOPs counts
        total_flops = encode_flops + decode_flops

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
            multi_topk_fvu,
            encode_flops,
            decode_flops,
            total_flops,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        assert self.W_dec.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )


# Allow for alternate naming conventions
Sae = SparseCoder
