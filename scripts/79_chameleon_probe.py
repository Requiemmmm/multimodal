#!/usr/bin/env python
"""Gateway probe for Chameleon-architecture checkpoints (Chameleon, Anole,
AA-chameleon, and other 65536-vocab single-stream models).

MORPH's attack needs a model that normally answers in text but retains image
generation in the SAME checkpoint and vocabulary.  The Chameleon family shares
Anole's exact structure:

    text tokens 0-8191 | image codes 8192-16383 | specials
    <racm3:break> (BOI) = 8197      <eoss> (EOI) = 8196

so "crossing the gateway" means emitting token 8197 instead of continuing text.

This probe measures, per checkpoint, whether the gateway exists at all:

  ordinary_q     plain text question      -> text expected (gateway shut)
  image_request  explicit draw request    -> does it emit 8197?

A checkpoint that never emits 8197 has no elicitable image mode, and cannot
host the attack no matter how it is prompted.  Only the transformer is needed
for this; rendering an image additionally requires the VQ-GAN decoder, which
the HF-format mirrors omit (the raw-format ones ship tokenizer/vqgan.ckpt).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Verified against Anole's own token sets (the Chameleon family shares this
# vocabulary exactly).  Note image tokens live at 4..8195, NOT above the BOI --
# an earlier version of this probe assumed 8192..16383 and silently scored zero.
BOS_ID, EOS_ID, PAD_ID = 0, 2, 1
BOI_ID, EOI_ID, EOT_ID = 8197, 8196, 8710
IMAGE_LO, IMAGE_HI = 4, 8195

ORDINARY_Q = "Explain in one sentence why strawberries are red."
IMAGE_REQUEST = "Draw a red apple on a wooden table."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    # Token-id overrides so the same probe covers other single-stream families.
    # Liquid (Gemma backbone): --boi-id 7 --eoi-id 8 --image-lo 256000
    # --image-hi 264191 --prompt-format plain
    parser.add_argument("--boi-id", type=int, default=BOI_ID)
    parser.add_argument("--eoi-id", type=int, default=EOI_ID)
    parser.add_argument("--image-lo", type=int, default=IMAGE_LO)
    parser.add_argument("--image-hi", type=int, default=IMAGE_HI)
    parser.add_argument("--prompt-format", choices=["chameleon_ui", "plain"],
                        default="chameleon_ui",
                        help="chameleon_ui = [BOS] + text + [EOT] (Anole's "
                             "tokens_from_ui); plain = raw text only")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    import torch
    from transformers import ChameleonForConditionalGeneration, ChameleonProcessor

    from transformers import AutoModelForCausalLM, AutoProcessor
    try:
        processor = ChameleonProcessor.from_pretrained(args.model_path)
        model = ChameleonForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="cuda")
    except Exception:
        # Non-Chameleon single-stream families (e.g. Liquid on a Gemma backbone).
        from transformers import GemmaForCausalLM, GemmaTokenizerFast
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = GemmaForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="cuda")
        processor.tokenizer = GemmaTokenizerFast.from_pretrained(args.model_path)
    model.eval()

    conditions = [("ordinary_q", ORDINARY_Q), ("image_request", IMAGE_REQUEST)]
    records = []
    for name, question in conditions:
        torch.manual_seed(args.seed)
        # Chameleon ships NO chat template, so build the official UI form
        # directly: [BOS] + text + [EOT]  (this is exactly Anole's
        # tokens_from_ui with a single text turn and the <END-OF-TURN> sentinel).
        # Using a generic <|im_start|> template here produced garbage ids and a
        # spurious BOI->EOI->EOS collapse in an earlier run.
        text_ids = processor.tokenizer(question)["input_ids"]
        if args.prompt_format == "chameleon_ui":
            ui_ids = [BOS_ID] + [t for t in text_ids
                                 if t not in (BOS_ID, EOS_ID)] + [EOT_ID]
        else:
            ui_ids = [t for t in text_ids if t not in (BOS_ID, EOS_ID)]
        inputs = {"input_ids": torch.tensor([ui_ids], dtype=torch.long,
                                            device="cuda")}
        prompt_len = len(ui_ids)
        with torch.no_grad():
            # Take pad/eos from the tokenizer: they differ per family (Gemma
            # pad=0/eos=1 vs Chameleon pad=1/eos=2) and a wrong eos silently
            # truncates generation.
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=1.0,
                pad_token_id=getattr(processor.tokenizer, "pad_token_id", None) or PAD_ID,
                eos_token_id=getattr(processor.tokenizer, "eos_token_id", None) or EOS_ID)
        new_ids = generated[0][prompt_len:].tolist()
        n_boi = sum(1 for t in new_ids if t == args.boi_id)
        n_eoi = sum(1 for t in new_ids if t == args.eoi_id)
        n_visual = sum(1 for t in new_ids if args.image_lo <= t <= args.image_hi)
        # Text tokens sit ABOVE the image range in this vocabulary, so keep
        # everything that is not an image code (specials get dropped by decode).
        text = processor.tokenizer.decode(
            [t for t in new_ids if not (args.image_lo <= t <= args.image_hi)],
            skip_special_tokens=True)
        record = {
            "kind": "chameleon_probe", "condition": name, "question": question,
            "seed": args.seed, "n_new_tokens": len(new_ids),
            "n_boi": n_boi, "n_eoi": n_eoi, "n_visual_tokens": n_visual,
            "emitted_image_gateway": n_boi > 0,
            "decoded_text": text[:300],
            "first_new_ids": new_ids[:24],
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    (output / "metrics.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(json.dumps({
        "kind": "chameleon_probe_summary",
        "model": args.model_path,
        "gateway_by_condition": {r["condition"]: r["emitted_image_gateway"]
                                 for r in records},
    }, indent=2))


if __name__ == "__main__":
    main()
