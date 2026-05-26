from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

SHORT_CONTEXT = 4
MID_CONTEXT = 32
LONG_CONTEXT = 128


@dataclass(frozen=True)
class ModelSize:
    d_model: int
    n_heads: int
    ff_dim: int
    n_layers: int


def normalize_hidden(hidden: Tensor) -> Tensor:
    return F.normalize(hidden.float(), dim=-1).to(hidden.dtype)


def tied_logits(hidden: Tensor, embedding: nn.Embedding, *, temperature: float, normalize: bool) -> Tensor:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if normalize:
        hidden = normalize_hidden(hidden)
        embedding_weight = F.normalize(embedding.weight.float(), dim=-1).to(embedding.weight.dtype)
        return F.linear(hidden, embedding_weight) / temperature
    return F.linear(hidden, embedding.weight) / temperature


def normalized_target_embeddings(embedding: nn.Embedding, targets: Tensor) -> Tensor:
    return F.normalize(embedding.weight[targets].float(), dim=-1).to(embedding.weight.dtype)


def interior_local_loss(
    hidden: Tensor,
    *,
    embedding: nn.Embedding,
    targets: Tensor,
    temperature: float,
    normalize: bool,
    local_loss: str,
) -> Tensor:
    if local_loss == "ce":
        logits = tied_logits(hidden, embedding, temperature=temperature, normalize=normalize)
        return F.cross_entropy(logits, targets)

    target_embeddings = normalized_target_embeddings(embedding, targets)
    if local_loss == "cosine":
        return 1.0 - F.cosine_similarity(hidden.float(), target_embeddings.float(), dim=-1).mean()
    if local_loss == "l2":
        return F.mse_loss(hidden.float(), target_embeddings.float())
    raise ValueError(f"Unsupported local_loss: {local_loss}")


class CausalSelfAttention(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model} and {n_heads}.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, context_size, d_model = x.shape
        queries, keys, values = self.qkv(x).chunk(3, dim=-1)

        def reshape_heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch_size, context_size, self.n_heads, self.head_dim).transpose(1, 2)

        queries = reshape_heads(queries)
        keys = reshape_heads(keys)
        values = reshape_heads(values)
        attended = F.scaled_dot_product_attention(queries, keys, values, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, context_size, d_model)
        return self.out_proj(attended)


class FeedForward(nn.Module):
    def __init__(self, *, d_model: int, ff_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, ff_dim)
        self.out_proj = nn.Linear(ff_dim, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_proj(F.gelu(self.in_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, ff_dim=ff_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SequenceEncoder(nn.Module):
    def __init__(self, *, context_size: int, d_model: int, n_heads: int, ff_dim: int, n_layers: int) -> None:
        super().__init__()
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        self.context_size = context_size
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model=d_model, n_heads=n_heads, ff_dim=ff_dim) for _ in range(n_layers)
        )

    def forward(self, inputs: Tensor, token_embedding: nn.Embedding) -> Tensor:
        if inputs.shape[1] != self.context_size:
            raise ValueError(f"Expected context {self.context_size}, got {inputs.shape[1]}")
        positions = torch.arange(self.context_size, device=inputs.device)
        hidden = token_embedding(inputs) + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class TiedReadoutModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        model_size: ModelSize,
        use_mid: bool,
        use_long: bool,
        temperature: float,
        lateral_scale: float,
        normalize: bool,
        local_loss: str,
    ) -> None:
        super().__init__()
        self.model_size = model_size
        self.temperature = temperature
        self.lateral_scale = lateral_scale
        self.normalize = normalize
        self.local_loss = local_loss
        self.token_embedding = nn.Embedding(vocab_size, model_size.d_model)
        self.output_block = SequenceEncoder(
            context_size=SHORT_CONTEXT,
            d_model=model_size.d_model,
            n_heads=model_size.n_heads,
            ff_dim=model_size.ff_dim,
            n_layers=model_size.n_layers,
        )
        self.mid_block = (
            SequenceEncoder(
                context_size=MID_CONTEXT,
                d_model=model_size.d_model,
                n_heads=model_size.n_heads,
                ff_dim=model_size.ff_dim,
                n_layers=model_size.n_layers,
            )
            if use_mid
            else None
        )
        self.long_block = (
            SequenceEncoder(
                context_size=LONG_CONTEXT,
                d_model=model_size.d_model,
                n_heads=model_size.n_heads,
                ff_dim=model_size.ff_dim,
                n_layers=model_size.n_layers,
            )
            if use_long
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=self.model_size.d_model**-0.5)

    def block_last_hidden(self, block: SequenceEncoder, inputs: Tensor) -> Tensor:
        hidden = block(inputs, self.token_embedding)[:, -1, :]
        if self.normalize:
            return normalize_hidden(hidden)
        return hidden

    def block_last_hidden_only(self, block: SequenceEncoder, inputs: Tensor) -> Tensor:
        return self.block_last_hidden(block, inputs)

    def combine_hidden(
        self,
        output_hidden: Tensor,
        *,
        mid_hidden: Tensor | None = None,
        long_hidden: Tensor | None = None,
    ) -> Tensor:
        combined_hidden = output_hidden
        if mid_hidden is not None:
            combined_hidden = combined_hidden + self.lateral_scale * mid_hidden.detach()
        if long_hidden is not None:
            combined_hidden = combined_hidden + self.lateral_scale * long_hidden.detach()
        return combined_hidden

    def output_logits_from_hidden(
        self,
        output_hidden: Tensor,
        *,
        mid_hidden: Tensor | None = None,
        long_hidden: Tensor | None = None,
    ) -> Tensor:
        return tied_logits(
            self.combine_hidden(output_hidden, mid_hidden=mid_hidden, long_hidden=long_hidden),
            self.token_embedding,
            temperature=self.temperature,
            normalize=self.normalize,
        )

    def output_logits(
        self,
        short_inputs: Tensor,
        *,
        mid_hidden: Tensor | None = None,
        long_hidden: Tensor | None = None,
    ) -> Tensor:
        output_hidden = self.block_last_hidden(self.output_block, short_inputs)
        return self.output_logits_from_hidden(output_hidden, mid_hidden=mid_hidden, long_hidden=long_hidden)


__all__ = [
    "CausalSelfAttention",
    "FeedForward",
    "LONG_CONTEXT",
    "MID_CONTEXT",
    "ModelSize",
    "SHORT_CONTEXT",
    "SequenceEncoder",
    "TiedReadoutModel",
    "TransformerBlock",
    "interior_local_loss",
    "normalize_hidden",
    "normalized_target_embeddings",
    "tied_logits",
]
