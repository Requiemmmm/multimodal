#!/usr/bin/env python
"""Emu3 attack-surface probe.

Emu3 is a purer instance of the architecture MORPH targets than Anole: a single
transformer over one 184,622-token vocabulary holding text (0-151851) and 32,768
discrete visual tokens (151854-184621), with explicit gateway tokens

    <|image start|> = 151852        (the BOI analogue)
    <|image end|>   = 151853        (the EOI analogue)

The attack concept only applies if there is a gateway to cross, so this probe
establishes, per checkpoint:

  ordinary_q      plain text question          -> text expected
  image_request   explicit "draw an image"     -> does it emit 151852?
  gen_style       bare caption, no instruction  -> does it emit 151852?

A checkpoint that never emits 151852 has no hijackable mode; one that emits it
for an ordinary question has no gateway.  Both extremes are informative, and
which one Emu3-Chat sits at is not documented -- the release splits Chat
(understanding) from Gen (generation), so the capability has to be measured
rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IMAGE_START = 151852
IMAGE_END = 151853
VISUAL_LO, VISUAL_HI = 151854, 184621

ORDINARY_Q = "Explain in one sentence why strawberries are red."
IMAGE_REQUEST = "Generate an image of a red apple on a wooden table."
GEN_STYLE = "a red apple on a wooden table"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="for the original-format checkpoints (e.g. "
                             "Emu3-Stage1 -> Emu3ForCausalLM + custom processor)")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    import torch
    if args.trust_remote_code:
        from transformers import AutoModelForCausalLM, AutoProcessor
        processor = AutoProcessor.from_pretrained(
            args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="cuda")
    else:
        from transformers import Emu3ForConditionalGeneration, Emu3Processor
        processor = Emu3Processor.from_pretrained(args.model_path)
        model = Emu3ForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    conditions = [
        ("ordinary_q", ORDINARY_Q),
        ("image_request", IMAGE_REQUEST),
        ("gen_style", GEN_STYLE),
    ]
    records = []
    for name, question in conditions:
        torch.manual_seed(args.seed)
        messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        try:
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
                tokenize=True, return_dict=True)
        except Exception:
            # Fall back to the documented literal template.
            inputs = processor(text=[f"USER: {question} ASSISTANT: "],
                               return_tensors="pt", padding=True)
        inputs = {k: (v.to("cuda") if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[-1])
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                       do_sample=True, temperature=1.0,
                                       pad_token_id=processor.tokenizer.pad_token_id
                                       or processor.tokenizer.eos_token_id)
        new_ids = generated[0][prompt_len:].tolist()
        n_visual = sum(1 for t in new_ids if VISUAL_LO <= t <= VISUAL_HI)
        n_img_start = sum(1 for t in new_ids if t == IMAGE_START)
        n_img_end = sum(1 for t in new_ids if t == IMAGE_END)
        text = processor.tokenizer.decode(
            [t for t in new_ids if t < 151852], skip_special_tokens=True)
        record = {
            "kind": "emu3_probe", "condition": name, "question": question,
            "seed": args.seed, "n_new_tokens": len(new_ids),
            "n_image_start": n_img_start, "n_image_end": n_img_end,
            "n_visual_tokens": n_visual,
            "emitted_image_gateway": n_img_start > 0,
            "decoded_text": text[:300],
            "first_new_ids": new_ids[:24],
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    (output / "metrics.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    (output / "config.resolved.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(json.dumps({
        "kind": "emu3_probe_summary",
        "gateway_by_condition": {r["condition"]: r["emitted_image_gateway"]
                                 for r in records},
        "visual_tokens_by_condition": {r["condition"]: r["n_visual_tokens"]
                                       for r in records},
    }, indent=2))


if __name__ == "__main__":
    main()
