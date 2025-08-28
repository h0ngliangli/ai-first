"""
Tiny GPT language model (self-contained, CPU-friendly)

Features:
- Character-level tokenizer
- Causal self-attention (multi-head)
- Transformer blocks with GELU MLP
- Simple training loop on toy text
- Autoregressive text generation

Run: python temp4.py
"""

import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------- Tokenizer (char-level) ----------------------
class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s: str):
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


# ---------------------- GPT Model ----------------------
@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size

        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # causal mask registered as buffer for speed
        mask = torch.tril(torch.ones(block_size, block_size)).view(
            1, 1, block_size, block_size
        )
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q = (
            self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        )  # (B, nh, T, hd)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.resid_drop(self.out_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(
            cfg.n_embd, cfg.n_head, cfg.block_size, cfg.dropout
        )
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg.n_embd, cfg.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "Sequence length exceeds block_size"
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)

        x = self.tok_emb(idx) + self.pos_emb(pos)  # (B, T, C)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab)

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
    ):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # (B, vocab)
            if top_k is not None:
                k = min(max(int(top_k), 0), logits.size(-1))
                if k > 0 and k < logits.size(-1):
                    v, ix = torch.topk(logits, k)
                    mask = torch.full_like(logits, float("-inf"))
                    mask.scatter_(1, ix, v)
                    logits = mask
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ---------------------- Data utils ----------------------
def make_toy_text():
    return (
        "To be, or not to be, that is the question:\n"
        "Whether 'tis nobler in the mind to suffer\n"
        "The slings and arrows of outrageous fortune,\n"
    )


def make_dataset(tokenizer: CharTokenizer, block_size: int, device: str):
    text = make_toy_text()
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # build simple contiguous sequences for next-token prediction
    X, Y = [], []
    for i in range(0, len(ids) - block_size - 1):
        X.append(ids[i : i + block_size])
        Y.append(ids[i + 1 : i + 1 + block_size])
    X = torch.stack(X)
    Y = torch.stack(Y)
    # tiny dataloader-like tensors
    return X.to(device), Y.to(device)


def iterate_minibatches(X, Y, batch_size: int):
    N = X.size(0)
    idx = torch.randperm(N)
    for i in range(0, N, batch_size):
        j = idx[i : i + batch_size]
        yield X[j], Y[j]


# ---------------------- Train & Demo ----------------------
def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer & config
    raw_text = make_toy_text()
    tokenizer = CharTokenizer(raw_text)
    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.1,
    )

    # data
    X, Y = make_dataset(tokenizer, cfg.block_size, device)

    # model
    model = TinyGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # train a few steps (quick demo)
    model.train()
    steps = 100
    batch_size = 32
    for step in range(steps):
        for xb, yb in iterate_minibatches(X, Y, batch_size):
            optimizer.zero_grad()
            _, loss = model(xb, yb)
            loss.backward()
            optimizer.step()
        if (step + 1) % 20 == 0:
            print(f"step {step+1}/{steps} loss: {loss.item():.4f}")

    # generate
    model.eval()
    context = "To be, or not to be, "
    ctx_ids = torch.tensor([tokenizer.encode(context)], dtype=torch.long, device=device)
    out_ids = model.generate(ctx_ids, max_new_tokens=120, temperature=0.8, top_k=50)
    print("\n--- sample ---")
    print(tokenizer.decode(out_ids[0].tolist()))


if __name__ == "__main__":
    main()
