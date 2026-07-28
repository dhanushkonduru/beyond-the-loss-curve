"""
=============================================================
 FORGET-ME-NOT : Machine Unlearning Capstone
 PHASE 1 — Model Architecture (GPT-2 style, from scratch)
=============================================================

Builds a decoder-only transformer in pure PyTorch:
  - token embeddings + learned positional embeddings
  - multi-head causal self-attention
  - feed-forward (MLP) layers with GELU
  - pre-LayerNorm (GPT-2 style)
  - weight-tied output head (lm_head shares weights with token embedding)

Tokenizer: HuggingFace GPT-2 tokenizer (50257 tokens).

Two size presets:
  - "small-50m"  : ~51M params  -> RECOMMENDED for free T4 (trains fast)
  - "gpt2-117m"  : ~124M params -> full GPT-2 small size (slower, still OK on T4)

Run this file directly to sanity-check the model:
  python model.py
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------
# Device detection (works on Colab T4, Kaggle, Mac, CPU)
# -------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():          # Apple Silicon (local testing)
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print(f"[Phase 1] Using device: {DEVICE}")


# -------------------------------------------------------------
# Model configuration
# -------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 50257     # GPT-2 tokenizer vocabulary
    block_size: int = 512       # max sequence length (TOFU QA pairs are short, 512 is plenty)
    n_layer: int = 8            # number of transformer blocks
    n_head: int = 8             # attention heads
    n_embd: int = 512           # embedding / hidden dimension
    dropout: float = 0.1


PRESETS = {
    # ~51M params — trains on TOFU in well under an hour on a T4
    "small-50m": GPTConfig(n_layer=8, n_head=8, n_embd=512, block_size=512),
    # ~124M params — the classic GPT-2 small shape ("117M")
    "gpt2-117m": GPTConfig(n_layer=12, n_head=12, n_embd=768, block_size=512),
}


# -------------------------------------------------------------
# Multi-head causal self-attention
# -------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # Q, K, V projections for all heads, computed in one matmul
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection back to the residual stream
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # causal mask so a token can only attend to tokens before it
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # project to Q, K, V and split heads: (B, n_head, T, head_dim)
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            # fused attention kernel (fast path on modern PyTorch / T4)
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # manual attention (readable reference implementation)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # merge heads back: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


# -------------------------------------------------------------
# Feed-forward network (position-wise MLP)
# -------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


# -------------------------------------------------------------
# Transformer block (pre-LayerNorm, GPT-2 style)
# -------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # attention + residual
        x = x + self.mlp(self.ln_2(x))    # feed-forward + residual
        return x


# -------------------------------------------------------------
# The full GPT model
# -------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
            wpe=nn.Embedding(config.block_size, config.n_embd),   # positional embeddings
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),                     # final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying: output head shares the token embedding matrix
        # (saves ~25M params and improves small-model quality)
        self.lm_head.weight = self.transformer.wte.weight

        # GPT-2 style initialization
        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 paper trick)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"[Phase 1] Model built: {self.num_params()/1e6:.1f}M parameters "
              f"({config.n_layer} layers, {config.n_head} heads, dim {config.n_embd})")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) token ids
        targets: (B, T) token ids shifted internally for next-token prediction.
                 Positions with -100 are ignored in the loss (used later to
                 mask out question tokens / padding).
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"

        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # next-token prediction: logits at position t predict token t+1
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=50, eos_token_id=None):
        """Autoregressive sampling. Used for testing and for the Phase 5 audit."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            # a heavily-unlearned model can emit NaN/Inf logits — degrade to
            # greedy argmax on the finite entries instead of crashing sampling
            if not torch.isfinite(logits).all():
                logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
                idx = torch.cat([idx, logits.argmax(dim=-1, keepdim=True)], dim=1)
                continue
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        return idx


# -------------------------------------------------------------
# Sanity test — run this file directly
# -------------------------------------------------------------
if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    print("\n[Phase 1] Loading GPT-2 tokenizer from HuggingFace...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token; reuse EOS

    print("[Phase 1] Building model (preset: small-50m)...")
    config = PRESETS["small-50m"]
    model = GPT(config).to(DEVICE)

    # --- forward pass test ---
    text = "Question: Who is Jaime Vasquez? Answer:"
    ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    logits, loss = model(ids, targets=ids)
    print(f"[Phase 1] Forward pass OK — input {tuple(ids.shape)}, logits {tuple(logits.shape)}")
    print(f"[Phase 1] Untrained loss: {loss.item():.3f} "
          f"(expected ≈ ln(50257) ≈ {math.log(config.vocab_size):.3f})")

    # --- generation test (untrained -> gibberish, that's correct) ---
    out = model.generate(ids, max_new_tokens=20, eos_token_id=tokenizer.eos_token_id)
    print(f"[Phase 1] Sample generation (untrained, should be gibberish):")
    print("  ", tokenizer.decode(out[0]))

    print("\n[Phase 1] ✅ Architecture complete. Next: Phase 2 (TOFU dataset loading).")
