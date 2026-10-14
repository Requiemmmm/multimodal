# MORPH 2027 research project

MORPH studies modality and content control in early-fusion unified-vocabulary multimodal models.
The project uses two linked interfaces:

- **Soft control**: continuous suffix embeddings, used to measure the model's internal capability upper bound across models and prompts.
- **Discrete control**: ordinary text-token suffixes, used to test native reachability of the modality gateway and the visual plan.

The current working thesis is **route–content dissociation**: a text suffix can open an image gateway more readily than it can preserve an attacker-specified visual plan. This is a live research hypothesis; new discrete routes remain open.

## Canonical project surfaces

- [`protocols/experiment_protocol.md`](protocols/experiment_protocol.md): frozen evaluation protocol and evidence rules.
- [`benchmarks/morph_benchmark.yaml`](benchmarks/morph_benchmark.yaml): victim prompts, target concepts, splits, and annotations.
- [`experiments/experiment_matrix.md`](experiments/experiment_matrix.md): main table, ablations, robustness, and priorities.
- [`baselines/baseline_matrix.md`](baselines/baseline_matrix.md): implemented, required, and non-comparable baselines.
- [`handoff/AGENT_TASKS.md`](handoff/AGENT_TASKS.md): executable tasks for future agents.
- [`paper/latex/main.tex`](paper/latex/main.tex): single canonical LaTeX draft directory.
- [`PROJECT_PLAN.md`](PROJECT_PLAN.md): fixed venue, scope, evidence gates, and execution order.

Historical runs remain in `runs/STATUS.md` and the external runtime tree. Do not rewrite or delete them.
