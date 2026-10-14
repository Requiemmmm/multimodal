#!/usr/bin/env python
"""Four-arm last-shot trainer for the hard-text content flip (ICLR sprint).

Arms (composable):
  A  victim-context projection-aware training: --train-prompts-file with
     --quant-penalty (same machinery as script 37's lambda=5 arm).
  B  BOI event objective, replacing preamble NLL:
       --objective boi-margin : log-sum-exp of the processed BOI logit margin
                                over teacher-forced preamble positions;
       --objective boi-rwr    : reward-weighted regression on rollouts that
                                actually hit BOI (P(tau_BOI <= T)), with
                                margin fallback when no rollout hits.
  C  conditional contrastive content: --content-contrastive W adds
     W * (L_target - L_victim) on the first K visual tokens under the CFG
     image policy, forced-BOI conditioned (teacher trajectory carries the
     target tokens; the victim teacher's tokens are scored what-if).
  D  straight-through hard forward: --ste projects each slot to its nearest
     text token for every forward and passes gradients through identity.

--objective route-nll reproduces the script 37 objective (arm A baseline).
"""
from __future__ import annotations

import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d08 = _load("distill08", str(PROJECT_ROOT / "scripts/08_distill_anole_soft_route.py"))
p12 = _load("ladder12", str(PROJECT_ROOT / "scripts/12_anole_gateway_patch_ladder.py"))
p14 = _load("adam14", str(PROJECT_ROOT / "scripts/14_adam_soft_route.py"))

from morph.decoding.processed_policy import apply_repetition_penalty, masked_logprobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suffix-length", type=int, default=64)
    parser.add_argument("--steps-route", type=int, default=30)
    parser.add_argument("--steps-content", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--objective", choices=["route-nll", "boi-margin", "boi-rwr"],
                        default="route-nll")
    parser.add_argument("--teacher-text", required=True,
                        help="target concept instruction (its freeform "
                             "trajectory provides preamble + BOI + visual)")
    parser.add_argument("--content-teacher-texts", nargs=2, default=None,
                        metavar=("TARGET", "VICTIM"),
                        help="teacher pair for --content-contrastive; default "
                             "= [--teacher-text, 'Draw a strawberry']")
    parser.add_argument("--train-prompts-file", required=True,
                        help="JSON list of victim prompt texts (ensemble)")
    parser.add_argument("--content-visual-steps", type=int, default=32)
    parser.add_argument("--content-contrastive", type=float, default=0.0)
    parser.add_argument("--quant-penalty", type=float, default=0.0)
    parser.add_argument("--ste", action="store_true",
                        help="hard forward (per-slot NN projection) with "
                             "straight-through gradients")
    parser.add_argument("--init-suffix", default=None)
    parser.add_argument("--guidance-text", type=float, default=3.0)
    parser.add_argument("--guidance-image", type=float, default=1.2)
    parser.add_argument("--image-temp", type=float, default=0.7)
    parser.add_argument("--image-top-p", type=float, default=0.9)
    parser.add_argument("--rwr-trajectories", type=int, default=4)
    parser.add_argument("--rwr-max-steps", type=int, default=64)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    sys.path.insert(0, str(Path(args.code_root)))

    import torch
    from chameleon.inference import transformer
    from chameleon.inference.loader import load_model
    from chameleon.inference.chameleon import TokenManager
    from morph.adapters.anole import load_anole_token_sets

    torch.set_default_device("cuda")
    model_path = Path(args.model_path)
    tokenizer_path = model_path / "tokenizer/text_tokenizer.json"
    token_sets = load_anole_token_sets(str(tokenizer_path))
    token_manager = TokenManager(
        str(tokenizer_path), str(model_path / "tokenizer/vqgan.yaml"),
        str(model_path / "tokenizer/vqgan.ckpt"), device="cuda",
    )
    model = load_model(str(model_path / "models/7b"))
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Anole model did not initialize on CUDA")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    boi_id, eos_id, eoi_id = token_sets.boi_id, token_sets.eos_id, token_sets.eoi_id
    support = d08.legal_mask(token_sets)
    text_support = p12.build_mask(token_sets, token_sets.text_step_support)
    image_mask = p12.build_mask(token_sets, token_sets.image_token_ids)

    prompt_texts = json.loads(Path(args.train_prompts_file).read_text())
    prompt_ids_list = [token_manager.tokens_from_ui([
        {"type": "text", "value": text},
        {"type": "sentinel", "value": "<END-OF-TURN>"},
    ]) for text in prompt_texts]

    def teacher_trajectory(text, k_visual):
        """Freeform teacher rollout: preamble + BOI + first k_visual tokens."""
        teacher_ids = token_manager.tokens_from_ui([
            {"type": "text", "value": text},
            {"type": "sentinel", "value": "<END-OF-TURN>"},
        ])
        mask = p12.build_mask(token_sets, token_sets.text_step_support)
        branch = p12.Branch(model, transformer, teacher_ids,
                            len(teacher_ids) + 256 + args.content_visual_steps + 8)
        preamble = []
        for _step in range(128):
            history = torch.tensor(branch.history, dtype=torch.long, device="cuda")
            policy = p12.text_policy(branch.logits, history, mask, 1.2, 0.9)
            token = p12.sample_token(policy)
            preamble.append(token)
            branch.advance(token)
            if token == boi_id:
                break
            if token == eos_id:
                raise RuntimeError(f"teacher hit EOS without BOI: {text}")
        image_cond_allowed = set(token_sets.image_token_ids) | {token_sets.bos_id, boi_id, eoi_id}
        imgcond_ids = [t for t in teacher_ids if t in image_cond_allowed]
        if not imgcond_ids or imgcond_ids[-1] != boi_id:
            imgcond_ids.append(boi_id)
        uncond_ids = [token_sets.bos_id, boi_id]
        aux = args.content_visual_steps + 8
        imgcond = p12.Branch(model, transformer, imgcond_ids, aux)
        uncond = p12.Branch(model, transformer, uncond_ids, aux)
        visual = []
        for _ in range(k_visual):
            mixed = (uncond.logits
                     + args.guidance_image * (imgcond.logits - uncond.logits)
                     + args.guidance_text * (branch.logits - imgcond.logits))
            policy = p12.image_policy(mixed, image_mask, args.image_temp, args.image_top_p)
            token = p12.sample_token(policy)
            visual.append(token)
            branch.advance(token)
            imgcond.advance(token)
            uncond.advance(token)
        return preamble, visual

    target_preamble, target_visual = teacher_trajectory(args.teacher_text, args.content_visual_steps)
    content_texts = args.content_teacher_texts or [args.teacher_text, "Draw a strawberry"]
    _vp, victim_visual = teacher_trajectory(content_texts[1], args.content_visual_steps)
    target_ids = list(target_preamble)  # preamble ends with BOI
    (output / "teachers.json").write_text(json.dumps({
        "target_text": args.teacher_text,
        "victim_text": content_texts[1],
        "target_preamble": target_ids,
        "target_visual": target_visual,
        "victim_visual": victim_visual,
    }, indent=2) + "\n")

    vocab = model.tok_embeddings.weight.detach().float()

    if args.init_suffix:
        payload = torch.load(args.init_suffix, map_location="cpu", weights_only=True)
        suffix = payload["suffix"].float().to("cuda")
        if suffix.shape[0] != args.suffix_length:
            raise SystemExit(
                f"init suffix has {suffix.shape[0]} slots, expected {args.suffix_length}")
        suffix = suffix.detach().requires_grad_(True)
    else:
        torch.manual_seed(args.init_seed)
        mean_embedding = vocab.mean(dim=0)
        suffix = (mean_embedding.repeat(args.suffix_length, 1)
                  + 0.01 * torch.randn(args.suffix_length, mean_embedding.numel(), device="cuda"))
        suffix = suffix.detach().requires_grad_(True)

    def projected_hard(suffix_soft):
        with torch.no_grad():
            vocab_n = torch.nn.functional.normalize(vocab, dim=-1)
            cos = torch.nn.functional.normalize(suffix_soft.detach(), dim=-1) @ vocab_n.T
            ids = cos.argmax(dim=-1)
        return vocab[ids]

    def suffix_forward_tensor(suffix_soft):
        if args.ste:
            hard = projected_hard(suffix_soft)
            return hard.detach() + (suffix_soft - suffix_soft.detach())
        return suffix_soft

    def route_nll(suffix_use):
        losses = [
            p14.differentiable_route_nll(
                model, transformer, token_sets,
                prompt_ids, suffix_use, target_ids, support)
            for prompt_ids in prompt_ids_list
        ]
        return torch.stack(losses).mean()

    def boi_margin(suffix_use):
        """-logsumexp of processed BOI margins along the teacher preamble."""
        margins = []
        for prompt_ids in prompt_ids_list:
            embeds = d08.soft_input_embeddings(
                model, prompt_ids[:-1], token_sets.eot_id, suffix_use, target_ids[:-1])
            logits = p14.feasibility.exact_forward(model, transformer, embeds)
            first_prediction = len(prompt_ids) + suffix_use.shape[0] - 1
            history = list(prompt_ids)
            for offset in range(len(target_ids) - 1):
                past = torch.tensor(history, dtype=torch.long, device="cuda")
                policy = masked_logprobs(
                    apply_repetition_penalty(logits[first_prediction + offset], past, penalty=1.2),
                    support)
                margin = policy[boi_id] - policy.max()
                margins.append(margin)
                history.append(target_ids[offset])
        stacked = torch.stack(margins)
        return -torch.logsumexp(stacked, dim=0)

    def boi_rwr(suffix_use):
        """RWR on trajectories that hit BOI; margin fallback otherwise."""
        hit_preludes = []
        for trajectory in range(args.rwr_trajectories):
            prompt_ids = prompt_ids_list[trajectory % len(prompt_ids_list)]
            torch.manual_seed(args.init_seed * 1000 + trajectory)
            context = d08.SoftTextContext(
                model, transformer, token_sets, prompt_ids, suffix_use,
                args.rwr_max_steps + 8)
            prelude = []
            for _step in range(args.rwr_max_steps):
                history = torch.tensor(context.history, dtype=torch.long, device="cuda")
                policy = p12.text_policy(context.logits, history, text_support, 1.2, 0.9)
                token = p12.sample_token(policy)
                if token == boi_id:
                    hit_preludes.append((prompt_ids, prelude))
                    break
                if token == eos_id:
                    break
                prelude.append(token)
                context.advance(token)
        if not hit_preludes:
            return boi_margin(suffix_use), 0
        losses = []
        for prompt_ids, prelude in hit_preludes:
            target_with_boi = prelude + [boi_id]
            embeds = d08.soft_input_embeddings(
                model, prompt_ids[:-1], token_sets.eot_id, suffix_use,
                target_with_boi[:-1])
            logits = p14.feasibility.exact_forward(model, transformer, embeds)
            first_prediction = len(prompt_ids) + suffix_use.shape[0] - 1
            past = torch.tensor(prompt_ids + prelude, dtype=torch.long, device="cuda")
            policy = masked_logprobs(
                apply_repetition_penalty(
                    logits[first_prediction + len(prelude)], past, penalty=1.2),
                support)
            losses.append(-policy[boi_id])
        return torch.stack(losses).mean(), len(hit_preludes)

    def contrastive_content(suffix_use):
        """Forced-BOI: one forward over the target teacher's visual tokens;
        score target and victim tokens at every visual position under CFG."""
        if len(target_visual) < 2:
            return torch.zeros((), device="cuda")
        prompt_losses = []
        for prompt_ids in prompt_ids_list:
            continuation = [boi_id] + target_visual[:-1]
            embeds = d08.soft_input_embeddings(
                model, prompt_ids[:-1], token_sets.eot_id, suffix_use, continuation)
            logits = p14.feasibility.exact_forward(model, transformer, embeds)
            first_prediction = len(prompt_ids) + suffix_use.shape[0] - 1
            imgcond = p12.Branch(model, transformer, [boi_id],
                                 len(target_visual) + 8)
            uncond = p12.Branch(model, transformer, [token_sets.bos_id, boi_id],
                                len(target_visual) + 8)
            deltas = []
            for k in range(len(target_visual)):
                mixed = (uncond.logits
                         + args.guidance_image * (imgcond.logits - uncond.logits)
                         + args.guidance_text * (logits[first_prediction + 1 + k] - imgcond.logits))
                policy = masked_logprobs(mixed, image_mask)
                deltas.append(policy[target_visual[k]] - policy[victim_visual[k]])
                imgcond.advance(target_visual[k])
                uncond.advance(target_visual[k])
            prompt_losses.append(-torch.stack(deltas).mean())
        return torch.stack(prompt_losses).mean()

    optimizer = torch.optim.Adam([suffix], lr=args.learning_rate)
    metrics_path = output / "metrics.jsonl"

    def append(record):
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        print(json.dumps(record), flush=True)

    def optimize_step(loss, step, phase, extra=None):
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(suffix.grad).all():
            raise RuntimeError("non-finite suffix gradient")
        optimizer.step()
        log = {"kind": f"lastshot_{phase}", "step": step,
               "loss": float(loss.detach()),
               "seconds": round(time.time() - t0, 2)}
        if extra:
            log.update({k: v for k, v in extra.items() if k != "loss"})
        append(log)

    t0 = time.time()
    hits_total = 0
    for step in range(args.steps_route + args.steps_content):
        suffix_use = suffix_forward_tensor(suffix)
        if args.objective == "route-nll":
            route_loss = route_nll(suffix_use)
        elif args.objective == "boi-margin":
            route_loss = boi_margin(suffix_use)
        else:
            route_loss, hits = boi_rwr(suffix_use)
            hits_total += hits
        loss = route_loss
        record = {"route_loss": float(route_loss.detach())}
        if args.objective == "boi-rwr":
            record["hits"] = hits_total
        if args.quant_penalty:
            vocab_norm = torch.nn.functional.normalize(vocab, dim=-1)
            slot_cos = torch.nn.functional.normalize(suffix, dim=-1) @ vocab_norm.T
            quant_loss = -slot_cos.max(dim=-1).values.mean()
            record["max_cos_mean"] = float(-quant_loss.detach())
            loss = loss + args.quant_penalty * quant_loss
        if step >= args.steps_route and args.content_contrastive:
            content_loss = contrastive_content(suffix_use)
            record["content_loss"] = float(content_loss.detach())
            loss = loss + args.content_contrastive * content_loss
        optimize_step(loss, step, "route" if step < args.steps_route else "content",
                      extra=record)
    torch.save({"suffix": suffix.detach().cpu(), "suffix_length": args.suffix_length},
               output / "soft_suffix.pt")
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    append({"kind": "lastshot_done", "seconds": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
