#!/usr/bin/env python
"""Check whether an exact-forward analytic backward is feasible for the M-step.

Previous sessions classified differentiable optimization as a DEGRADATION:
rope_padded rejects gradients, and a PyTorch surrogate backward (explicit
softmax attention) took over five minutes per elite preflight.  This script
tests a different construction instead:

- forward: official ``rope_padded`` + the xformers flash attention kernel
  (``memory_efficient_attention``, the autograd wrapper of the same kernel the
  decoder calls raw) with the official per-layer cache, bit-identical values
  to the deployed decoder;
- backward: native xformers flash attention backward plus an analytic inverse
  RoPE.  The attention's gradients w.r.t. the K/V cache are routed into the
  custom RoPE Function by returning the cache tensors as identity outputs, so
  autograd delivers them in topological order (no .grad probing).

Checks:
1. confirm the known rope_padded backward rejection;
2. finite-difference check of the new path's suffix gradient (ground truth);
3. cross-check against the old surrogate path on a short sequence;
4. wall-clock forward+backward for a route-elite length and a full-image length.
"""
import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import argparse
import json
import sys
import time
from pathlib import Path



import torch
from xformers.ops import fmha, rope_padded
from xformers.ops.rmsnorm import rms_norm


# --------------------------------------------------------------------------
# Custom autograd Function: exact official RMSNorm forward, analytic backward.
# --------------------------------------------------------------------------

class RMSNormWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        with torch.no_grad():
            out = rms_norm(x, weight, eps)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_y):
        x, w = ctx.saved_tensors
        xf, wf, gyf = x.float(), w.float(), grad_y.float()
        dim = x.shape[-1]
        var = xf.square().mean(-1, keepdim=True)
        r = torch.sqrt(var + ctx.eps)
        t = (gyf * wf * xf).sum(-1, keepdim=True)
        # y = x*w/r with r^2 = mean(x^2)+eps:
        #   grad_x = (g*w - x*t/(dim*(var+eps))) / r,  t = sum(g*w*x)
        grad_x = (gyf * wf - t * xf / (dim * (var + ctx.eps))) / r
        grad_w = (gyf * xf / r).sum(tuple(range(x.dim() - 1)))
        return grad_x.to(x.dtype), grad_w.to(w.dtype), None


# --------------------------------------------------------------------------
# Custom autograd Function: exact official RoPE forward, analytic backward.
# --------------------------------------------------------------------------

class RopePaddedWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xq, xk, xv, cache_k, cache_v, attn_bias, theta):
        # Full-sequence prefill only: every cache position in [k_start, k_end)
        # is written by this very call, so the whole slice maps to xk/xv.
        q_start = int(attn_bias.q_seqinfo.seqstart_py[0])
        q_end = int(attn_bias.q_seqinfo.seqstart_py[-1])
        k_start = int(attn_bias.k_seqinfo.seqstart_py[0])
        k_end = int(attn_bias.k_seqinfo.seqstart_py[-1])
        if q_start != k_start or (q_end - q_start) != (k_end - k_start):
            raise NotImplementedError(
                "cache-gradient rope backward only supports full-sequence prefill"
            )
        ctx.q_start, ctx.q_len = q_start, q_end - q_start
        ctx.theta = float(theta)
        with torch.no_grad():
            out_q = rope_padded(
                xq=xq, xk=xk, xv=xv, cache_k=cache_k, cache_v=cache_v,
                attn_bias=attn_bias, theta=theta,
            )
        # Identity outputs: attention's KV gradients arrive here as backward
        # inputs instead of landing on detached cache leaves.
        return out_q, cache_k, cache_v

    @staticmethod
    def backward(ctx, grad_out_q, grad_cache_k, grad_cache_v):
        positions = torch.arange(
            ctx.q_len, device=grad_out_q.device, dtype=torch.float32
        )
        grad_xq = _inverse_rope(grad_out_q, positions, ctx.theta)
        grad_xk = _inverse_rope(
            grad_cache_k[:, ctx.q_start:ctx.q_start + ctx.q_len],
            positions, ctx.theta,
        )
        grad_xv = grad_cache_v[:, ctx.q_start:ctx.q_start + ctx.q_len]
        return grad_xq, grad_xk, grad_xv, None, None, None, None


def _inverse_rope(x, positions, theta):
    """Undo the kernel's RoPE: freqs = pos * theta^(-2i/dim), adjacent pairs."""
    dim = x.shape[-1]
    powers = torch.arange(0, dim // 2, device=x.device, dtype=torch.float32)
    freqs = positions[:, None] * torch.pow(theta, -2.0 * powers / dim)
    cos = torch.cos(freqs)[None, :, None, :]
    sin = torch.sin(freqs)[None, :, None, :]
    real = x[..., 0::2]
    imag = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = real * cos + imag * sin
    out[..., 1::2] = imag * cos - real * sin
    return out


# --------------------------------------------------------------------------
# Exact official layer forward with the differentiable RoPE wrapper.
# --------------------------------------------------------------------------

def differentiable_layer_forward(layer, hidden, cache, padded_bias, causal_bias):
    """Mirror chameleon/inference/transformer.py TransformerBlock.forward.

    The RoPE/cache write uses the official offset-padded bias; the attention
    then runs on the GQA-expanded BHMK layout with a plain causal bias, which
    the flash kernel proves bit-identical to the official BMGHK padded path
    for a full-window prefill while remaining differentiable."""
    attn = layer.attention
    normed = RMSNormWithGrad.apply(hidden, layer.attention_norm.weight,
                                   layer.attention_norm.eps)
    xqkv = attn.wqkv(normed)
    nq, nkv, hd = attn.n_local_heads, attn.n_local_kv_heads, attn.head_dim
    length = xqkv.shape[0]
    xq = xqkv[:, :nq * hd]
    xk, xv = xqkv[:, nq * hd:].chunk(2, 1)
    if attn.qk_normalization:
        xq = attn.q_normalization(xq.view(-1, nq, hd)).view(-1, nq * hd)
        xk = attn.k_normalization(xk.view(-1, nkv, hd)).view(-1, nkv * hd)
    output_shape = xq.shape
    xq = xq.view(1, length, nq, hd)
    xk = xk.view(1, length, nkv, hd)
    xv = xv.view(1, length, nkv, hd)
    cache_k, cache_v = cache
    xq_roped, cache_k, cache_v = RopePaddedWithGrad.apply(
        xq, xk, xv, cache_k, cache_v, padded_bias, attn.rope_theta
    )
    heads_per_group = nq // nkv
    cache_k_g = cache_k.unsqueeze(3).expand(-1, -1, -1, heads_per_group, -1).reshape(1, length, nq, hd)
    cache_v_g = cache_v.unsqueeze(3).expand(-1, -1, -1, heads_per_group, -1).reshape(1, length, nq, hd)
    output = fmha.memory_efficient_attention(xq_roped, cache_k_g, cache_v_g, attn_bias=causal_bias)
    output = attn.wo(output.reshape(output_shape))
    h = hidden + output
    return h + layer.feed_forward(
        RMSNormWithGrad.apply(h, layer.ffn_norm.weight, layer.ffn_norm.eps)
    )


def make_biases(transformer, length):
    from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
    padded = transformer.AttnBias.from_seqlens(
        q_seqlen=[length], kv_seqlen=[length], kv_padding=length
    )
    causal = BlockDiagonalCausalMask.from_seqlens([length])
    return padded, causal


def exact_forward(model, transformer, embeddings, return_hidden=False):
    """Layer-wise official forward; differentiable when inputs require grad.

    ``return_hidden=True`` additionally returns the pre-final-norm residual
    stream (the gateway-state coordinate used by the P1 ladder)."""
    length = embeddings.shape[0]
    cache = transformer.make_cache(
        model.args, length=length, device=embeddings.device, dtype=embeddings.dtype
    )
    padded_bias, causal_bias = make_biases(transformer, length)
    hidden = embeddings.to(dtype=next(model.parameters()).dtype)
    for index, layer in enumerate(model.layers):
        hidden = differentiable_layer_forward(
            layer, hidden, cache[index], padded_bias, causal_bias
        )
    logits = model.output(
        RMSNormWithGrad.apply(hidden, model.norm.weight, model.norm.eps)
    ).float()
    return (logits, hidden) if return_hidden else logits


def official_forward(model, transformer, embeddings):
    """Bit-reference: official layer calls under no_grad (script 08 path)."""
    length = embeddings.shape[0]
    cache = transformer.make_cache(
        model.args, length=length, device=embeddings.device, dtype=embeddings.dtype
    )
    bias = transformer.AttnBias.from_seqlens(
        q_seqlen=[length], kv_seqlen=[length], kv_padding=length
    )
    hidden = embeddings.to(dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        for index, layer in enumerate(model.layers):
            hidden = layer(hidden, cache[index], bias)
        return model.output(model.norm(hidden)).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", default=os.environ.get("ANOLE_MODEL_PATH", "/srv/morph/models/anole-7b-v0.1"))
    parser.add_argument("--elites", default=None)
    parser.add_argument("--long-trajectory", default=None)
    parser.add_argument("--suffix-length", type=int, default=16)
    args = parser.parse_args()

    torch.set_default_device("cuda")
    import importlib.util
    from chameleon.inference.loader import load_model
    from chameleon.inference import transformer
    from chameleon.inference.chameleon import TokenManager

    print("loading model...", flush=True)
    model = load_model(str(Path(args.model_root) / "models/7b"))
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("model did not initialize on CUDA")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer_path = Path(args.model_root) / "tokenizer/text_tokenizer.json"
    token_manager = TokenManager(
        str(tokenizer_path), str(Path(args.model_root) / "tokenizer/vqgan.yaml"),
        str(Path(args.model_root) / "tokenizer/vqgan.ckpt"), device="cuda",
    )
    prompt_ids = token_manager.tokens_from_ui([
        {"type": "text", "value": "Explain in one sentence why apples are a popular fruit."},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ])
    elites = json.loads(Path(args.elites).read_text())["red_apple"]
    route_target = torch.tensor(elites[0]["target_ids"], device="cuda")
    print(f"prompt length {len(prompt_ids)}, route elite length {len(route_target)}", flush=True)

    # --- 1. confirm the known rope_padded rejection -------------------------
    q = torch.zeros(1, 8, 32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros(1, 8, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.zeros_like(k)
    ck = torch.zeros(1, 8, 8, 128, device="cuda", dtype=torch.bfloat16)
    cv = torch.zeros_like(ck)
    bias = transformer.AttnBias.from_seqlens(q_seqlen=[8], kv_seqlen=[8], kv_padding=8)
    try:
        rope_padded(xq=q, xk=k, xv=v, cache_k=ck, cache_v=cv, attn_bias=bias, theta=10000.0)
        rope_rejection = False
    except ValueError as err:
        rope_rejection = f"{type(err).__name__}: {err}"
    print(f"[1] rope_padded under grad -> {rope_rejection}", flush=True)

    # --- 2/3. gradient checks on the real route objective --------------------
    torch.manual_seed(0)
    mean_embedding = model.tok_embeddings.weight.detach().float().mean(dim=0)
    suffix = (mean_embedding.repeat(args.suffix_length, 1)
              + 0.01 * torch.randn(args.suffix_length, mean_embedding.numel(), device="cuda"))
    suffix = suffix.detach().requires_grad_(True)
    prefix_ids = torch.tensor(prompt_ids, device="cuda")

    def build_embeddings(suf, target_ids):
        embedded = model.tok_embeddings(torch.cat([prefix_ids, target_ids]))
        return torch.cat((embedded[:len(prefix_ids)], suf.to(embedded.dtype),
                          embedded[len(prefix_ids):]), dim=0)

    def route_like_nll(logits, target_ids):
        # logits at position prefix_len + suffix_len - 1 + j predicts target[j]
        start = len(prefix_ids) + args.suffix_length - 1
        rows = logits[start:start + len(target_ids)]
        return torch.nn.functional.cross_entropy(rows, target_ids.to(rows.device))

    # forward equivalence against the official decoder path
    with torch.no_grad():
        logits_official = official_forward(
            model, transformer, build_embeddings(suffix.detach(), route_target)
        )
        logits_new_ng = exact_forward(
            model, transformer, build_embeddings(suffix.detach(), route_target)
        )
    print(f"[0] forward equivalence (L={len(route_target) + args.suffix_length}): "
          f"max abs diff {(logits_new_ng - logits_official).abs().max().item():.3e}", flush=True)

    print("short-sequence backward check...", flush=True)
    embeddings = build_embeddings(suffix, route_target)
    logits = exact_forward(model, transformer, embeddings)
    loss = route_like_nll(logits, route_target)
    t0 = time.time()
    loss.backward()
    grad_new = suffix.grad.detach().clone()
    print(f"[2] analytic backward ok, {time.time()-t0:.2f}s; grad norm {grad_new.norm():.4f}", flush=True)

    # finite difference on one coordinate
    idx = (int(torch.randint(0, args.suffix_length, (1,))),
           int(torch.randint(0, mean_embedding.numel(), (1,))))
    eps = 0.1
    losses = []
    for sign in (+1.0, -1.0):
        pert = suffix.detach().clone()
        pert[idx] += sign * eps
        with torch.no_grad():
            l = route_like_nll(exact_forward(model, transformer, build_embeddings(pert, route_target)), route_target)
        losses.append(float(l))
    fd_grad = (losses[0] - losses[1]) / (2 * eps)
    rel_err = abs(fd_grad - grad_new[idx].item()) / max(abs(fd_grad), 1e-8)
    print(f"[3] FD check: fd {fd_grad:.6f} vs analytic {grad_new[idx].item():.6f} "
          f"(rel err {rel_err:.4f})", flush=True)

    # The old P0 surrogate path cannot be cross-checked here: its RMSNorm
    # calls sit outside its custom attention Function, so under no_grad they
    # detach and no gradient can reach the suffix (and with grad enabled the
    # xformers RMSNorm raises).  The finite-difference check above is the
    # ground truth; repeat it on a second coordinate for confidence.
    idx2 = (int(torch.randint(0, args.suffix_length, (1,))),
            int(torch.randint(0, mean_embedding.numel(), (1,))))
    losses2 = []
    for sign in (+1.0, -1.0):
        pert = suffix.detach().clone()
        pert[idx2] += sign * eps
        with torch.no_grad():
            l = route_like_nll(exact_forward(model, transformer, build_embeddings(pert, route_target)), route_target)
        losses2.append(float(l))
    fd_grad2 = (losses2[0] - losses2[1]) / (2 * eps)
    rel_err2 = abs(fd_grad2 - grad_new[idx2].item()) / max(abs(fd_grad2), 1e-8)
    print(f"[4] FD check 2: fd {fd_grad2:.6f} vs analytic {grad_new[idx2].item():.6f} "
          f"(rel err {rel_err2:.4f})", flush=True)

    # --- 5. timing ------------------------------------------------------------
    long_ids = None
    tokens = json.loads(Path(args.long_trajectory).read_text())
    boi = int(json.loads(Path(tokenizer_path).read_text())["model"]["vocab"]["<racm3:break>"])
    if boi in tokens:
        i = tokens.index(boi)
        long_ids = torch.tensor(tokens[: i + 1025], device="cuda")
    for name, ids in (("route-elite", route_target), ("full-image", long_ids)):
        if ids is None:
            print(f"[5] {name}: skipped (no BOI in trajectory)")
            continue
        with torch.no_grad():
            ref = official_forward(model, transformer, build_embeddings(suffix.detach(), ids))
            new = exact_forward(model, transformer, build_embeddings(suffix.detach(), ids))
        print(f"[5] {name} forward equivalence (L={len(ids)}): "
              f"max abs diff {(new - ref).abs().max().item():.3e}", flush=True)
        times = []
        for rep in range(3):
            suffix.grad = None
            emb = build_embeddings(suffix, ids)
            t0 = time.time()
            l = route_like_nll(exact_forward(model, transformer, emb), ids)
            l.backward()
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"[5] {name} (L={len(ids)}): fwd+bwd {min(times):.2f}s "
              f"(times {[f'{t:.2f}' for t in times]}), peak alloc {mem:.2f}GB", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
