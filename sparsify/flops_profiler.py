"""
FLOPs profiler for measuring computational complexity during training.
"""

from collections import defaultdict
from contextlib import contextmanager
from typing import Dict

import torch
import torch.nn as nn


class FLOPsProfiler:
    """Context manager for profiling FLOPs during model execution."""

    def __init__(self):
        self.flops_count = defaultdict(int)
        self.hooks = []
        self.enabled = False

    def _count_linear_flops(self, module, input, output):
        """Count FLOPs for linear layers."""
        if not self.enabled:
            return

        if isinstance(module, nn.Linear):
            # For linear layer: input_size * output_size * batch_size
            batch_size = (
                input[0].shape[0] if isinstance(input, tuple) else input.shape[0]
            )
            input_size = module.in_features
            output_size = module.out_features

            # Forward pass: input_size * output_size * batch_size
            flops = input_size * output_size * batch_size
            self.flops_count["linear_forward"] += flops

            # Backward pass (approximate): 2x forward pass
            self.flops_count["linear_backward"] += 2 * flops

    def _count_matmul_flops(self, module, input, output):
        """Count FLOPs for matrix multiplications."""
        if not self.enabled:
            return

        # This is a simplified count - in practice, you'd want to be more specific
        # about which operations are matmuls vs other operations
        pass

    def _count_activation_flops(self, module, input, output):
        """Count FLOPs for activation functions."""
        if not self.enabled:
            return

        if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
            # Activation functions typically have minimal FLOPs
            # but we count the element-wise operations
            if isinstance(input, tuple):
                input_tensor = input[0]
            else:
                input_tensor = input

            num_elements = input_tensor.numel()
            self.flops_count["activation"] += num_elements

    def _count_topk_flops(self, input_size: int, k: int, batch_size: int):
        """Count FLOPs for top-k operations."""
        if not self.enabled:
            return

        # Top-k typically requires O(n log k) operations, but we approximate as O(n)
        # where n is the input size
        flops = input_size * batch_size
        self.flops_count["topk"] += flops

    def _count_embedding_flops(self, module, input, output):
        """Count FLOPs for embedding operations."""
        if not self.enabled:
            return

        if isinstance(module, nn.Embedding):
            # Embedding lookup is typically O(1) per element, 
            # but we count the output size
            if isinstance(input, tuple):
                input_tensor = input[0]
            else:
                input_tensor = input

            batch_size = input_tensor.numel()
            embedding_dim = module.embedding_dim
            flops = batch_size * embedding_dim
            self.flops_count["embedding"] += flops

    def register_hooks(self, model: nn.Module):
        """Register hooks on the model to count FLOPs."""
        self.hooks = []

        for module in model.modules():
            # Register hooks for different types of operations
            hook = module.register_forward_hook(self._count_linear_flops)
            self.hooks.append(hook)

            hook = module.register_forward_hook(self._count_activation_flops)
            self.hooks.append(hook)

            hook = module.register_forward_hook(self._count_embedding_flops)
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def reset(self):
        """Reset the FLOPs counter."""
        self.flops_count.clear()

    def get_flops(self) -> Dict[str, int]:
        """Get the current FLOPs count."""
        return dict(self.flops_count)

    def get_total_flops(self) -> int:
        """Get the total FLOPs count."""
        return sum(self.flops_count.values())

    @contextmanager
    def profile(self, model: nn.Module):
        """Context manager for profiling FLOPs."""
        self.enabled = True
        self.reset()
        self.register_hooks(model)

        try:
            yield self
        finally:
            self.remove_hooks()
            self.enabled = False


class SparseCoderFLOPsProfiler:
    """Specialized FLOPs profiler for SparseCoder components."""

    def __init__(self):
        self.encoding_flops = 0
        self.decoding_flops = 0
        self.total_flops = 0
        self.enabled = False

    def count_encoding_flops(
        self,
        input_tensor: torch.Tensor,
        encoder_weight: torch.Tensor,
        encoder_bias: torch.Tensor,
        k: int,
    ):
        """Count FLOPs for the encoding step."""
        if not self.enabled:
            return

        batch_size, input_dim = input_tensor.shape
        latent_dim = encoder_weight.shape[0]

        # Linear transformation: input_dim * latent_dim * batch_size
        linear_flops = input_dim * latent_dim * batch_size

        # ReLU activation: latent_dim * batch_size (element-wise)
        relu_flops = latent_dim * batch_size

        # Top-k selection: approximate as latent_dim * batch_size
        topk_flops = latent_dim * batch_size

        self.encoding_flops = linear_flops + relu_flops + topk_flops

    def count_decoding_flops(
        self,
        top_acts: torch.Tensor,
        top_indices: torch.Tensor,
        decoder_weight: torch.Tensor,
        output_dim: int,
    ):
        """Count FLOPs for the decoding step."""
        if not self.enabled:
            return

        batch_size = top_acts.shape[0]
        k = top_acts.shape[1]

        # Decoder matrix multiplication: k * output_dim * batch_size
        # This is an approximation since we're only using top-k features
        self.decoding_flops = k * output_dim * batch_size

    def count_auxk_flops(
        self, dead_mask: torch.Tensor, k_aux: int, output_dim: int, batch_size: int
    ):
        """Count FLOPs for AuxK loss computation."""
        if not self.enabled:
            return

        # AuxK involves additional top-k selection and decoding
        auxk_flops = k_aux * output_dim * batch_size
        return auxk_flops

    def reset(self):
        """Reset all FLOPs counters."""
        self.encoding_flops = 0
        self.decoding_flops = 0
        self.total_flops = 0

    def get_total_flops(self) -> int:
        """Get total FLOPs for the sparse coder."""
        return self.encoding_flops + self.decoding_flops

    @contextmanager
    def profile(self):
        """Context manager for profiling sparse coder FLOPs."""
        self.enabled = True
        self.reset()

        try:
            yield self
        finally:
            self.enabled = False


def count_model_flops(model: nn.Module, input_shape: tuple) -> Dict[str, int]:
    """Count FLOPs for a model with given input shape."""
    profiler = FLOPsProfiler()

    with profiler.profile(model):
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()

        # Forward pass
        with torch.no_grad():
            _ = model(dummy_input)

    return profiler.get_flops()
