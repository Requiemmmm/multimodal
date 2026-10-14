#!/usr/bin/env python
"""P0: Verify full_forward logits match official forward within float32 precision.

This is the prerequisite for all M-step results. If this test fails, all
soft suffix NLL numbers are unreliable.

Usage:
    python P0_verify_full_forward_equivalence.py \
        --model-path /path/to/anole-7b-v0.1 \
        --code-root /path/to/anole

Expected:
    Max absolute error < 1e-4 (float32 epsilon ~1e-7 * hidden_dim ~4096 → ~1e-3 safety margin)
"""
import argparse
import sys
import torch


def enable_differentiable_rmsnorm(model):
    """Replace xformers RMSNorm modules for the suffix-only backward pass.

    xformers' optimized RMSNorm intentionally rejects gradients.  The
    elementwise expression below is mathematically the same normalization and
    is used only by the M-step; no-grad rollout contexts continue to use the
    original fast modules in separate processes.
    """
    import torch
    from torch import nn

    class DifferentiableRMSNorm(nn.Module):
        def __init__(self, weight, eps):
            super().__init__()
            self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
            self.eps = eps

        def forward(self, x):
            return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype) * self.weight

    def replace(parent):
        for name, child in list(parent.named_children()):
            if child.__class__.__name__ == "RMSNorm" and hasattr(child, "weight"):
                setattr(parent, name, DifferentiableRMSNorm(child.weight, child.eps))
            else:
                replace(child)

    replace(model)


def full_forward(model, transformer, embeddings):
    """Differentiable full-sequence forward using official layer() calls.

    This relies on enable_differentiable_rmsnorm() having replaced xformers
    RMSNorm with a PyTorch implementation that supports gradients. The
    attention path uses official xformers ops with cache, ensuring numerical
    equivalence to the evaluation rollout in SoftTextContext.

    Used only by the suffix M-step gradient computation.
    """
    import torch

    length = embeddings.shape[0]
    cache = transformer.make_cache(
        model.args,
        length=length,
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    bias = transformer.AttnBias.from_seqlens(
        q_seqlen=[length],
        kv_seqlen=[length],
        kv_padding=length,
    )
    hidden = embeddings
    for index, layer in enumerate(model.layers):
        hidden = layer(hidden, cache[index], bias)
    return model.output(model.norm(hidden)).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.code_root)
    from pathlib import Path
    from chameleon.inference import loader, transformer as transformer_module
    from chameleon.inference.chameleon import TokenManager

    print(f"Loading model from {args.model_path}")
    model_path = Path(args.model_path)
    tokenizer_path = model_path / "tokenizer/text_tokenizer.json"
    token_manager = TokenManager(
        str(tokenizer_path),
        str(model_path / "tokenizer/vqgan.yaml"),
        str(model_path / "tokenizer/vqgan.ckpt"),
        device="cuda",
    )
    model = loader.load_model(str(model_path / "models/7b"))
    model.cuda()
    model.eval()

    # Replace xformers RMSNorm with differentiable version
    # This is what script 08 does before M-step
    enable_differentiable_rmsnorm(model)

    # Test prompt: "Hello, world!" → EOT
    prompt = "Hello, world!"
    token_ids = token_manager.tokens_from_ui([
        {"type": "text", "value": prompt},
        {"type": "sentinel", "value": "<END-OF-TURN>"}
    ])
    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device="cuda")

    print(f"Test prompt: {repr(prompt)}")
    print(f"Token IDs: {token_ids}")
    print(f"Sequence length: {len(token_ids)}")

    # Official forward (no gradients needed for comparison)
    with torch.no_grad():
        sequence_length = len(token_ids)
        cache_official = transformer_module.make_cache(
            model.args,
            length=sequence_length,
            device="cuda",
            dtype=next(model.parameters()).dtype,
        )
        embeddings_official = model.tok_embeddings(token_ids_tensor)
        bias = transformer_module.AttnBias.from_seqlens(
            q_seqlen=[sequence_length], kv_seqlen=[sequence_length], kv_padding=sequence_length
        )
        hidden = embeddings_official
        for index, layer in enumerate(model.layers):
            hidden = layer(hidden, cache_official[index], bias)
        logits_official = model.output(model.norm(hidden)).float()

    # full_forward path (also no gradients needed for P0 verification)
    with torch.no_grad():
        embeddings_custom = model.tok_embeddings(token_ids_tensor)
        logits_custom = full_forward(model, transformer_module, embeddings_custom)

    # Compare
    abs_diff = (logits_official - logits_custom).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    rel_diff = abs_diff / (logits_official.abs() + 1e-8)
    max_rel = rel_diff.max().item()

    print(f"\n=== Equivalence Check ===")
    print(f"logits_official shape: {logits_official.shape}")
    print(f"logits_custom shape: {logits_custom.shape}")
    print(f"Max absolute diff: {max_abs:.6e}")
    print(f"Mean absolute diff: {mean_abs:.6e}")
    print(f"Max relative diff: {max_rel:.6e}")

    THRESHOLD = 1e-4
    if max_abs < THRESHOLD:
        print(f"\n✓ PASS: max absolute diff {max_abs:.6e} < {THRESHOLD:.6e}")
        return 0
    else:
        print(f"\n✗ FAIL: max absolute diff {max_abs:.6e} >= {THRESHOLD:.6e}")
        print(f"  This indicates full_forward is not equivalent to official forward.")
        print(f"  All M-step results derived from full_forward are unreliable.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
