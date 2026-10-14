# MORPH experiment matrix

## Main claims and tables

| Claim | Main experiment | Primary table/figure | Status |
|---|---|---|---|
| Soft suffixes expose internal route + content control | multi-model, multi-prompt soft evaluation | Table 1 / Fig. 1 | extend |
| Discrete suffixes can open the image gateway | objective ladder and fair baselines | Table 2 | partial/done on Anole |
| Route and visual-plan control separate | native content matrix with independent judging | Table 3 / Fig. 2 | expand |
| The separation has measurable mechanisms | sampling mass, prefix divergence, target diversity | Fig. 3 | partial |
| Findings transfer across model families | Anole, Lumina, one additional usable model | Table 4 | pending |

## Primary experiments

1. Soft suffix: Anole, Lumina, and a third usable model; 8–12 prompts, near/far targets, held-out paraphrases.
2. Discrete route ladder: no suffix, random, NN, quantization-aware, GCG-NLL, BOI-margin, BOI-CE, constrained joint search.
3. Discrete content: token, distribution, prototype/cluster, and on-policy behavioral objectives.
4. Independent evaluation: route, validity, content, joint success, readback, CLIP/noise controls, and human subset.

## Ablations

**已完成的 run**：
- **后缀长度消融**：run 124 (128-token, 路由率 81.46%, 内容 0/13), run 125 (256-token, 未恢复路由)
- **状态匹配路由**：run 18-20 (98.7% 状态匹配, 47/48 苹果), run 31 (跨请求失败, 0 目标内容)
- **四控制链**：run 126-129 (状态注入 0/15, 内容不在注入状态中)

**待补充的消融**：
- suffix length: 16/32/64 (TBD);
- route objective: margin vs CE and sampling-mass diagnostics (TBD);
- content objective: teacher-forced vs on-policy, token vs distribution vs prototype (TBD);
- route floor: unconstrained vs constrained Pareto search (P1 task #4);
- target trajectory count and semantic validation (TBD);
- projection fidelity and partial-slot quantization (TBD);
- prompt distance: near vs far (TBD);
- model and tokenizer family (TBD);
- official constrained decoder vs explicitly labeled diagnostic alternative (TBD);
- query budget and best-of-N as robustness, not a new method (TBD).

## Execution priority

P0: multi-prompt soft evaluation; formalize benchmark and judge; Lumina soft/native gateway matrix.
P1: fair GCG/I-GCG route baselines; constrained discrete search; codebook/prototype audit.
P2: prototype discrete search; on-policy Pareto search; third model.
P3: defense/output-mode gate and reverse-direction control.
