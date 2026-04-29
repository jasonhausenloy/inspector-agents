"""TinyGPT — character-level transformer trained on Tiny Shakespeare.

Sized for ~5-min wall time on M3 MPS at full instrumentation. Architecture is
standard decoder-only GPT with causal self-attention. Parameter count is
deliberately fixed by Config so the verifier's Kaplan FLOP estimator
(6 * n_params * tokens) matches what we declare on each log record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 65       # Tiny Shakespeare char-level
    block_size: int = 128      # context length (tokens)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = True

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


class CausalSelfAttention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        assert c.n_embd % c.n_head == 0
        self.c = c
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=c.bias)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=c.bias)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.c.n_head, self.c.head_dim).transpose(1, 2)
        k = k.view(B, T, self.c.n_head, self.c.head_dim).transpose(1, 2)
        v = v.view(B, T, self.c.n_head, self.c.head_dim).transpose(1, 2)
        # PyTorch SDPA is causal-masked + flash on MPS.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.fc = nn.Linear(c.n_embd, 4 * c.n_embd, bias=c.bias)
        self.proj = nn.Linear(4 * c.n_embd, c.n_embd, bias=c.bias)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.attn = CausalSelfAttention(c)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.mlp = MLP(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos_emb = nn.Embedding(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    @property
    def n_params(self) -> int:
        # Counts every parameter including the tied head (weight tying means
        # head.weight aliases tok_emb.weight; we count it once).
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.c.block_size, f"sequence length {T} > block_size {self.c.block_size}"
        pos = torch.arange(T, device=idx.device, dtype=torch.long)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.c.block_size else idx[:, -self.c.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -math.inf
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def kaplan_flops_per_step(n_params: int, batch_size: int, seq_len: int) -> float:
    """Kaplan-scaling FLOP estimator for one training step.

    Forward pass: 2 * N * tokens. Backward pass: 4 * N * tokens (roughly 2x
    forward). Total: 6 * N * tokens. This is the canonical estimate used by
    the verifier as a cross-check against declared flops.
    """
    return 6.0 * n_params * batch_size * seq_len


def kaplan_flops_per_inference_token(n_params: int) -> float:
    """For autoregressive inference, ~2 * N FLOPs per generated token."""
    return 2.0 * n_params
