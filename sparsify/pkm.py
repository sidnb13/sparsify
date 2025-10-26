import math

import torch
from einops import rearrange
from torch import einsum, nn

from .fused_encoder import EncoderOutput


def init_(t, dim=None):
    if dim is None:
        dim = t.shape[-1]
    std = 1.0 / math.sqrt(dim)
    return nn.init.normal_(t, mean=0, std=std)


class PKM(nn.Module):
    def __init__(
        self,
        dim,
        n,
        ctx_len=128,
        topk=32,
        input_dropout=0.0,
        query_dropout=0.0,
        pre_layernorm=False,
    ):
        super().__init__()

        assert dim % 2 == 0, "Dimension must be even."

        self.topk = topk
        self.num_keys = int(math.sqrt(n))
        self.ctx_len = ctx_len
        self.pre_layernorm = nn.LayerNorm(dim) if pre_layernorm else nn.Identity()
        self.norm = nn.LayerNorm(dim // 2)
        self.keys = nn.Parameter(torch.zeros(self.num_keys, 2, dim // 2))
        init_(self.keys)
        self.input_dropout = nn.Dropout(input_dropout)
        self.query_dropout = nn.Dropout(query_dropout)

    def forward(self, x, input_mask=None, **kwargs):
        x = self.pre_layernorm(x)
        x = self.input_dropout(x)
        # split out query heads
        queries = rearrange(x, "(b t) (p d) -> (b p) t d", p=2, t=self.ctx_len)
        # norm and dropout queries
        queries = self.norm(queries)
        queries = self.query_dropout(queries)
        # ready queries
        queries = rearrange(queries, "(b p) t d -> p b t d", p=2)
        # similarity to keys

        dots = einsum("p b t d, n p d -> b t p n", queries, self.keys)
        # topk scores
        scores, indices = dots.topk(k=self.topk, dim=-1)

        # scores are factorized
        (scores_x, scores_y), (indices_x, indices_y) = map(
            lambda t: t.chunk(2, dim=2), (scores, indices)
        )
        all_scores = rearrange(
            (
                rearrange(scores_x, "... k -> ... k 1")
                + rearrange(scores_y, "... k -> ... 1 k")
            ),
            "b t ... -> b t (...)",
        )
        all_indices = rearrange(
            (
                rearrange(indices_x, "... k -> ... k 1") * self.num_keys
                + rearrange(indices_y, "... k -> ... 1 k")
            ),
            "b t ... -> b t (...)",
        )

        final_topk, final_indices = all_scores.topk(self.topk, dim=-1)
        value_indices = all_indices.gather(-1, final_indices)

        # Return the final activations and indices
        return EncoderOutput(
            rearrange(final_topk, "b t ... -> (b t) ...", t=self.ctx_len),
            rearrange(value_indices, "b t ... -> (b t) ...", t=self.ctx_len),
            None,
        )
