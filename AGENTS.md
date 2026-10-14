# AGENTS.md

## Mission

Build a runnable MORPH research prototype for early-fusion unified-vocabulary multimodal models. Start with Lumina-mGPT-7B-768-Omni, a user-message text suffix, and MORPH-v0. The immediate goal is an end-to-end experimental loop, not a final paper claim.

Read `docs/MORPH_CODEX_NEW_SERVER_GUIDE.md`, the active config, and `runs/STATUS.md` before substantial work.

## Research behavior

- Distinguish “not yet demonstrated” from “false.” One failed seed, prompt, model, loss, or implementation does not invalidate the project.
- Classify issues as `BLOCKER`, `DEGRADATION`, or `OBSERVATION`. Only genuine blockers stop the current path; otherwise log the issue and continue with the next reversible experiment.
- Prefer the smallest end-to-end vertical slice over broad architecture, refactoring, or paper-level polish.
- Use the current project phase’s evidence standard. Smoke tests may use one prompt/seed and a simple evaluator; add rigorous statistics after a signal exists.
- Keep teacher-only, bridge, and student-only/native results separate. Never present forced BOI, mode flags, output prefill, or `alpha>0` as native success.
- Use benign image targets first. Do not overstate safety conclusions from models without image safety alignment.

## Engineering

- First support one model and one GPU path. Add abstractions only when the second adapter requires them.
- Model weights are shared between teacher and student; contexts/KV caches are separate.
- Rollouts use `torch.no_grad()`. M-step freezes model weights and optimizes suffix parameters only.
- Validate at external boundaries: CLI inputs, paths, model outputs, and saved artifacts. Do not duplicate internal checks or add broad silent fallbacks.
- Prefer one clear error over nested `try/except Exception` layers.
- If FlashAttention fails, use SDPA for the smoke test and record the fallback. Do not block the entire project on an optional accelerator.
- Run targeted tests after each change; reserve full image rollouts for integration checks.
- Do not modify unrelated code or create distributed/service infrastructure before the single-model loop works.

## Model and download integrity

- Record model Hub ID, snapshot/revision, code commit, file presence, total size, and one successful load/decode after initial download.
- Do not repeatedly hash multi-GB model shards. Re-hash only after suspected corruption, interrupted manual copies, or formal archival.
- Do not repeat full environment dumps or `pip freeze` per run. Freeze once after the environment works.
- Never delete or overwrite model weights or historical run artifacts without explicit user approval.

## Experiment rules

- Main position: user-message suffix. Code must allow prefix/infix later, but do not search all positions in the first prototype.
- First target: route-only soft suffix with semantics supplied by the normal prompt.
- Default MORPH-v0: same-support local policy bridge, quantile elite selection, replay, segmented route/visual NLL, soft suffix, periodic student-only evaluation, gradual alpha reduction.
- Apply common legality masks before the bridge and common sampling warpers after the bridge. Do not top-k teacher and student separately before interpolation.
- If teacher/student hard support differs, report it clearly and use a same-support context, finite-bias/context homotopy, or treat the privileged path as an oracle. Never add epsilon to illegal tokens to fake reachability.
- Separate route, validity, collapse, semantics, and joint success metrics.
- Save resolved config, seed, commits, metrics JSONL, trajectories, suffix state, and generated images for every substantial run.

## Workflow

1. State the objective, hypothesis, done condition, and files to change in a few lines.
2. Make the smallest coherent change.
3. Run the most relevant test or short experiment.
4. Save artifacts and update `runs/STATUS.md` with: observation, interpretation, and next action.
5. Continue through non-destructive local issues without asking for confirmation. Ask only before destructive actions, secret exposure, large paid external actions, or irreversible changes.

## Communication

- Be direct and technical. Do not repeat background or defensive caveats in every paragraph.
- Failure reports use: `phenomenon → likely layer → excluded causes → next experiment`.
- Do not spend a turn only discussing what could be done when a runnable file, command, test, or experiment can be produced.
- Keep progress updates concise and do not dump full logs, full configs, or large files unless requested.
