# MORPH project plan and decision record

## Fixed direction

- Venue family: machine learning and AI conferences. Draft for ICML/ICLR/NeurIPS/CVPR/ACL-style review; do not use a security-paper threat-model chapter as the organizing structure.
- Scientific scope: expand continuous soft-suffix evidence across prompts and models, while continuing ordinary-text discrete suffix search for both modality and content control.
- Central paper story: continuous control establishes an internal capability upper bound; discrete search measures native interface reachability; route and visual-plan/content control are evaluated as separate outcomes.
- MORPH is the framework. Optimizers and bridges are named variants (`MORPH-Soft`, `MORPH-NN`, `MORPH-QAware`, `MORPH-BOI-CE`, constrained joint search), and only an end-to-end validated variant may be called the canonical method.

## Internal research questions

1. Does continuous control generalize across prompts, concepts, and model families?
2. Which discrete objectives open the image gateway, and at what sampling mass?
3. Can ordinary text suffixes preserve attacker-specified visual content after routing?
4. Which trajectory, state, prototype, or exposure factors explain success and failure?
5. Which parts transfer across model/tokenizer families?

## Main claims require these evidence gates

- A multi-prompt soft matrix before claiming soft generality.
- A matched same-action-space baseline table before claiming discrete improvement.
- Native route/content/joint metrics with an independent content judge.
- Prompt-level statistics and held-out prompts; seeds are not the only unit of replication.
- At least two independent victim prompts and multiple seeds before a new discrete content route is promoted as a result.

## Canonical artifact rules

- Historical evidence: `runs/STATUS.md` and the external runtime run directories.
- New benchmark/protocol/experiment design: `research/morph2027/`.
- New LaTeX source and figures: `research/morph2027/paper/latex/` and `paper/figures/`.
- Never overwrite model weights or historical run artifacts.
- Every substantial run must be reproducible from a resolved config and leave JSONL metrics plus its suffix state.

## Near-term execution order

1. Finish the benchmark prompt files and independent content judge.
2. Run multi-prompt soft evaluation on Anole and Lumina.
3. Run Lumina native route/content smoke and establish its model-specific adapter.
4. Run matched GCG/I-GCG route baselines.
5. Audit visual codebook/latent prototypes before prototype-guided discrete search.
6. Use route-floor constrained search for every new joint/content objective.
7. Generate paper tables from metrics artifacts and keep LaTeX numbers out of handwritten prose.
