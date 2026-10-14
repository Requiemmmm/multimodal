#!/usr/bin/env python
"""Debug: Compare official vs full_forward layer-by-layer to find divergence point."""
import argparse
import sys
import torch


def enable_differentiable_rmsnorm(model):
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


def full_forward_with_intermediates(model, transformer, embeddings):
    """Return (final_logits, [(layer_idx, hidden_after_layer)])"""
    import torch
    import torch.nn.functional as F

    length = embeddings.shape[0]
    hidden = embeddings
    positions = torch.arange(length, device=hidden.device, dtype=torch.float32)
    intermediates = []

    for layer_idx, layer in enumerate(model.layers):
        attn = layer.attention
        x = layer.attention_norm(hidden)
        xqkv = attn.wqkv(x)
        q_width = attn.n_local_heads * attn.head_dim
        q = xqkv[:, :q_width].view(1, length, attn.n_local_heads, attn.head_dim).transpose(1, 2)
        kv = xqkv[:, q_width:].view(1, length, 2, attn.n_local_kv_heads, attn.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        if attn.qk_normalization:
            q, k = attn.q_normalization(q), attn.k_normalization(k)
        half = attn.head_dim // 2
        inv_freq = 1.0 / (attn.rope_theta ** (torch.arange(0, half, device=hidden.device, dtype=torch.float32) / half))
        angles = positions[:, None] * inv_freq[None, :]
        cos, sin = angles.cos().to(q.dtype)[None, None], angles.sin().to(q.dtype)[None, None]
        def rotate(z):
            first, second = z[..., :half], z[..., half:]
            return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)
        q, k = rotate(q), rotate(k)
        if k.shape[1] != q.shape[1]:
            repeats = q.shape[1] // k.shape[1]
            k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        hidden = hidden + attn.wo(attended.transpose(1, 2).reshape(length, q_width))
        hidden = hidden + layer.feed_forward(layer.ffn_norm(hidden))
        intermediates.append((layer_idx, hidden.clone()))

    logits = model.output(model.norm(hidden)).float()
    return logits, intermediates


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
    enable_differentiable_rmsnorm(model)

    prompt = "Hello, world!"
    token_ids = token_manager.tokens_from_ui([
        {"type": "text", "value": prompt},
        {"type": "sentinel", "value": "<END-OF-TURN>"}
    ])
    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device="cuda")

    print(f"Test prompt: {repr(prompt)}")
    print(f"Token IDs: {token_ids}")
    print(f"Sequence length: {len(token_ids)}\n")

    with torch.no_grad():
        # Official forward
        sequence_length = len(token_ids)
        cache_official = transformer_module.make_cache(
            model.args,
            length=sequence_length,
            device="cuda",
            dtype=next(model.parameters()).dtype,
        )
        embeddings = model.tok_embeddings(token_ids_tensor)
        bias = transformer_module.AttnBias.from_seqlens(
            q_seqlen=[sequence_length], kv_seqlen=[sequence_length], kv_padding=sequence_length
        )
        hidden_official = embeddings.clone()
        official_intermediates = []
        for index, layer in enumerate(model.layers):
            hidden_official = layer(hidden_official, cache_official[index], bias)
            official_intermediates.append((index, hidden_official.clone()))
        logits_official = model.output(model.norm(hidden_official)).float()

        # full_forward
        embeddings_custom = model.tok_embeddings(token_ids_tensor)
        logits_custom, custom_intermediates = full_forward_with_intermediates(model, transformer_module, embeddings_custom)

        # Compare layer by layer
        print("=== Layer-by-Layer Comparison ===")
        for (idx_off, hidden_off), (idx_cus, hidden_cus) in zip(official_intermediates, custom_intermediates):
            assert idx_off == idx_cus
            diff = (hidden_off - hidden_cus).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"Layer {idx_off:2d}: max_abs={max_diff:.6e}  mean_abs={mean_diff:.6e}")

        # Final logits
        logits_diff = (logits_official - logits_custom).abs()
        print(f"\nFinal logits: max_abs={logits_diff.max().item():.6e}  mean_abs={logits_diff.mean().item():.6e}")


if __name__ == "__main__":
    main()
