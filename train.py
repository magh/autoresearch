"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import re
import subprocess
import time
from functools import lru_cache
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Detect AMD ROCm vs NVIDIA CUDA
IS_ROCM = hasattr(torch.version, 'hip') and torch.version.hip is not None
if not IS_ROCM and "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

ATTENTION_BACKEND = os.environ.get("AUTORESEARCH_ATTENTION_BACKEND", "auto").strip().lower()
MODEL_COMPILE = os.environ.get("AUTORESEARCH_COMPILE", "auto").strip().lower()
LOGIT_CHUNK_SIZE = int(os.environ.get("AUTORESEARCH_LOGIT_CHUNK_SIZE", "0"))  # 0 = auto
DEVICE_INDEX = int(os.environ.get("AUTORESEARCH_DEVICE_INDEX", "-1"))  # -1 = auto
TIME_BUDGET_OVERRIDE = float(os.environ.get("AUTORESEARCH_TIME_BUDGET", "0"))  # 0 = fixed benchmark
EVAL_BATCH_SIZE = int(os.environ.get("AUTORESEARCH_EVAL_BATCH_SIZE", "0"))  # 0 = auto


@lru_cache(maxsize=1)
def _rocm_product_names():
    if not IS_ROCM:
        return {}
    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {}

    names = {}
    pattern = re.compile(r"GPU\[(\d+)\]\s*:\s*Card Series:\s*(.+)")
    for line in out.stdout.splitlines():
        match = pattern.search(line)
        if match:
            names[int(match.group(1))] = match.group(2).strip()
    return names


def get_device_name(device_index, props=None):
    if props is None:
        props = torch.cuda.get_device_properties(device_index)
    name = props.name
    if IS_ROCM:
        rocm_name = _rocm_product_names().get(device_index)
        if rocm_name:
            return rocm_name
    return name


def select_device_index():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device not available.")

    device_count = torch.cuda.device_count()
    if DEVICE_INDEX >= 0:
        if DEVICE_INDEX >= device_count:
            raise ValueError(
                f"AUTORESEARCH_DEVICE_INDEX={DEVICE_INDEX} is out of range for {device_count} visible devices."
            )
        props = torch.cuda.get_device_properties(DEVICE_INDEX)
        print(f"Using device {DEVICE_INDEX}: {get_device_name(DEVICE_INDEX, props)} ({getattr(props, 'gcnArchName', 'unknown')})")
        return DEVICE_INDEX

    candidates = []
    for idx in range(device_count):
        props = torch.cuda.get_device_properties(idx)
        name = props.name.lower()
        arch = getattr(props, "gcnArchName", "") or ""
        score = (
            "780m" in name or "graphics" in name,
            -getattr(props, "multi_processor_count", 0),
            -props.total_memory,
            idx,
        )
        candidates.append((score, idx, get_device_name(idx, props), arch))

    _, selected_idx, selected_name, selected_arch = min(candidates)
    print(f"Using device {selected_idx}: {selected_name} ({selected_arch or 'unknown'})")
    return selected_idx


DEVICE_INDEX = select_device_index()
torch.cuda.set_device(DEVICE_INDEX)
GPU_NAME = get_device_name(DEVICE_INDEX)

def _select_attention_backend():
    if ATTENTION_BACKEND not in {"auto", "fa3", "sdpa"}:
        raise ValueError(
            "AUTORESEARCH_ATTENTION_BACKEND must be one of: auto, fa3, sdpa"
        )

    if ATTENTION_BACKEND == "sdpa":
        print("Attention backend: PyTorch SDPA (forced by AUTORESEARCH_ATTENTION_BACKEND=sdpa)")
        return "sdpa", None

    if IS_ROCM:
        if ATTENTION_BACKEND == "fa3":
            raise RuntimeError("Flash Attention 3 is not supported on ROCm in this repo; use SDPA instead.")
        print("Attention backend: PyTorch SDPA (ROCm)")
        return "sdpa", None

    cap = torch.cuda.get_device_capability(DEVICE_INDEX)
    gpu_name = GPU_NAME
    wants_fa3 = ATTENTION_BACKEND == "fa3" or cap == (9, 0)
    if not wants_fa3:
        print(f"Attention backend: PyTorch SDPA (CUDA fallback for {gpu_name}, capability {cap})")
        return "sdpa", None

    from kernels import get_kernel

    try:
        repo = "varunneal/flash-attention-3"
        fa3 = get_kernel(repo).flash_attn_interface
        print(f"Attention backend: Flash Attention 3 via {repo} on {gpu_name}")
        return "fa3", fa3
    except Exception as exc:
        if ATTENTION_BACKEND == "fa3":
            raise RuntimeError(f"Failed to initialize Flash Attention 3: {exc}") from exc
        print(f"Attention backend: PyTorch SDPA (FA3 unavailable: {exc})")
        return "sdpa", None

ATTENTION_BACKEND, fa3 = _select_attention_backend()


def should_compile_model():
    if MODEL_COMPILE not in {"auto", "0", "1", "false", "true"}:
        raise ValueError("AUTORESEARCH_COMPILE must be one of: auto, 0, 1, false, true")
    if MODEL_COMPILE in {"0", "false"}:
        print("torch.compile: disabled by AUTORESEARCH_COMPILE")
        return False
    if MODEL_COMPILE in {"1", "true"}:
        print("torch.compile: enabled by AUTORESEARCH_COMPILE")
        return True
    enabled = ATTENTION_BACKEND == "fa3" and not IS_ROCM
    reason = "FA3 fast path" if enabled else "fallback backend"
    print(f"torch.compile: {'enabled' if enabled else 'disabled'} ({reason})")
    return enabled


USE_MODEL_COMPILE = should_compile_model()


def select_logit_chunk_size():
    if LOGIT_CHUNK_SIZE < 0:
        raise ValueError("AUTORESEARCH_LOGIT_CHUNK_SIZE must be >= 0")
    if LOGIT_CHUNK_SIZE > 0:
        print(f"Logit chunk size: {LOGIT_CHUNK_SIZE} (forced by AUTORESEARCH_LOGIT_CHUNK_SIZE)")
        return LOGIT_CHUNK_SIZE
    chunk_size = 0 if ATTENTION_BACKEND == "fa3" else 128
    if chunk_size > 0:
        print(f"Logit chunk size: {chunk_size} (auto-selected for fallback backend)")
    else:
        print("Logit chunk size: full sequence")
    return chunk_size


LOGIT_CHUNK_SIZE = select_logit_chunk_size()

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def scaled_dot_product_attention(q, k, v):
    q = q.transpose(1, 2)  # (B, T, H, D) -> (B, H, T, D)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return y.transpose(1, 2).contiguous()


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        if ATTENTION_BACKEND == "fa3":
            y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # SDPA doesn't support window_size, so SSSL pattern degrades to full
            # causal attention on layers using the fallback backend.
            y = scaled_dot_product_attention(q, k, v)
        y = y.view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        if targets is not None and LOGIT_CHUNK_SIZE > 0:
            loss_parts = []
            total_loss = None
            total_tokens = None
            for start in range(0, T, LOGIT_CHUNK_SIZE):
                end = min(start + LOGIT_CHUNK_SIZE, T)
                logits = self.lm_head(x[:, start:end]).float()
                logits = softcap * torch.tanh(logits / softcap)
                chunk_targets = targets[:, start:end].reshape(-1)
                chunk_logits = logits.reshape(-1, logits.size(-1))
                if reduction == 'none':
                    loss_parts.append(F.cross_entropy(
                        chunk_logits, chunk_targets, ignore_index=-1, reduction='none'
                    ))
                else:
                    chunk_loss = F.cross_entropy(
                        chunk_logits, chunk_targets, ignore_index=-1, reduction='sum'
                    )
                    valid_tokens = (chunk_targets != -1).sum()
                    total_loss = chunk_loss if total_loss is None else total_loss + chunk_loss
                    total_tokens = valid_tokens if total_tokens is None else total_tokens + valid_tokens
            if reduction == 'none':
                return torch.cat(loss_parts, dim=0)
            if reduction == 'sum':
                return total_loss
            if reduction == 'mean':
                return total_loss / total_tokens.clamp_min(1)
            raise ValueError(f"Unsupported reduction: {reduction}")

        logits = self.lm_head(x).float()
        logits = softcap * torch.tanh(logits / softcap)
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

_maybe_compile = torch.compile(dynamic=False, fullgraph=True) if not IS_ROCM else lambda fn: fn

@_maybe_compile
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    dtype = exp_avg.dtype
    exp_avg.lerp_(grad, (1 - beta1_t).to(dtype=dtype))
    exp_avg_sq.lerp_(grad.square(), (1 - beta2_t).to(dtype=dtype))
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@_maybe_compile
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), (1 - beta2).to(dtype=second_momentum_buffer.dtype))
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = int(os.environ.get("AUTORESEARCH_DEVICE_BATCH_SIZE", "0"))  # 0 = auto

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
# Peak BF16 FLOPS by GPU model (for MFU calculation)
_GPU_PEAK_FLOPS = {
    "H100":   989.5e12,
    "H200":   989.5e12,
    "A100":   312.0e12,
    "B200":   2250.0e12,
    # AMD Instinct
    "MI300X": 1307.4e12,
    "MI308X": 1307.4e12,
    "MI325X": 1307.4e12,
    "MI250X": 383.0e12,
    # AMD Radeon AI PRO
    # Uses AMD's advertised 191 TFLOPS FP16 matrix throughput as a BF16 proxy.
    "R9700": 191.0e12,
}

def _detect_peak_flops():
    gpu_name = GPU_NAME
    for key, flops in _GPU_PEAK_FLOPS.items():
        if key.lower() in gpu_name.lower():
            print(f"Detected GPU: {gpu_name} -> peak BF16 FLOPS: {flops:.1e}")
            return flops
    print(f"Warning: Unknown GPU '{gpu_name}', MFU reporting disabled")
    return None

PEAK_BF16_FLOPS = _detect_peak_flops()

TRAINING_TIME_BUDGET = TIME_BUDGET_OVERRIDE if TIME_BUDGET_OVERRIDE > 0 else TIME_BUDGET


def _select_device_batch_size():
    if DEVICE_BATCH_SIZE > 0:
        print(f"Device batch size: {DEVICE_BATCH_SIZE} (forced by AUTORESEARCH_DEVICE_BATCH_SIZE)")
        return DEVICE_BATCH_SIZE

    gpu_name = GPU_NAME.lower()
    datacenter_gpus = ("h100", "h200", "a100", "b200", "mi300", "mi308", "mi325", "mi250")
    if any(name in gpu_name for name in datacenter_gpus) and ATTENTION_BACKEND == "fa3":
        batch_size = 128
    elif any(name in gpu_name for name in datacenter_gpus):
        batch_size = 32
    elif ATTENTION_BACKEND == "fa3":
        batch_size = 32
    else:
        batch_size = 8
    print(f"Device batch size: {batch_size} (auto-selected for {GPU_NAME})")
    return batch_size


DEVICE_BATCH_SIZE = _select_device_batch_size()


def _select_eval_batch_size():
    if EVAL_BATCH_SIZE > 0:
        print(f"Eval batch size: {EVAL_BATCH_SIZE} (forced by AUTORESEARCH_EVAL_BATCH_SIZE)")
        return EVAL_BATCH_SIZE

    total_vram_gb = torch.cuda.get_device_properties(DEVICE_INDEX).total_memory / 1024 / 1024 / 1024
    if total_vram_gb >= 24:
        batch_size = max(DEVICE_BATCH_SIZE, 32)
    elif total_vram_gb >= 12:
        batch_size = max(DEVICE_BATCH_SIZE, 16)
    else:
        batch_size = DEVICE_BATCH_SIZE
    print(f"Eval batch size: {batch_size} (auto-selected for {GPU_NAME})")
    return batch_size


EVAL_BATCH_SIZE = _select_eval_batch_size()

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

if USE_MODEL_COMPILE:
    model = torch.compile(model, dynamic=False)
else:
    print("Model runs without torch.compile")

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TRAINING_TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TRAINING_TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / TRAINING_TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = None if PEAK_BF16_FLOPS is None else 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / PEAK_BF16_FLOPS
    remaining = max(0, TRAINING_TIME_BUDGET - total_training_time)
    mfu_text = "n/a" if mfu is None else f"{mfu:.1f}%"

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu_text} | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TRAINING_TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
eval_start = time.time()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, EVAL_BATCH_SIZE)
eval_seconds = time.time() - eval_start

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = None if PEAK_BF16_FLOPS is None or total_training_time <= 0 else (
    100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / PEAK_BF16_FLOPS
)
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"startup_seconds:  {startup_time:.1f}")
print(f"eval_seconds:     {eval_seconds:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {'n/a' if steady_state_mfu is None else f'{steady_state_mfu:.2f}'}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
print(f"eval_batch_size:  {EVAL_BATCH_SIZE}")
