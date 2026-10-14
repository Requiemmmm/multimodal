# MORPH 迁移证据审计

更新时间：2026-10-15  
范围：旧服务器 `/home/yding/workbar/morph-runtime` 与当前工作区 `/home/yding/workbar/guide`。  
规则：本文件只把可追溯 artifact 作为证据；没有原始日志、配置或数值的项目标为 `MISSING`。

## 结论

现有迁移手册的大方向正确：新服务器应先做 Anole 官方 decoder、native baseline、faithful post-BOI，再做 soft route 和 64-slot quantization-aware → projection → BOI-CE hard route。需要修正的是证据等级：

- `guide/` 不是 Git 仓库，因此当前代码没有可报告的 commit，也没有 Git patch。
- Anole/Lumina 外部仓库 commit、模型 revision、主要权重和 tokenizer/VQ hash 可以追溯。
- stock xformers 的 `rope_padded` backward 已有真实失败 traceback；不能说“xformers backward 可用”。`P0_backward_feasibility.py` 的 custom analytic path 需要在新服务器重新验收。
- 现有 E0 artifact 的 cache 分支等价为 0，但 finite-difference 误差很大，且多处 descent test 失败；这不是 suffix backward 已验证的证据。
- 历史 hard route 的 `41/80` 是 native evaluator 的结果；soft、teacher、bridge、forced BOI 和 GCG loss 必须分栏，不能混为 native 成功率。
- 旧 run 的 prompt map 有一部分来自 `~/.claude/jobs/...`，属于迁移缺失项；新服务器必须重建并把 map 写入 run artifact。

## 0. 如何把本审计交给新服务器

本文件不是“复制后即可执行的 shell script”。其中出现的 `/home/yding/workbar/morph-runtime/...`、`/home/yding/workbar/guide/...` 和 `~/.claude/jobs/...` 都是旧服务器 provenance，不能在新服务器直接使用。

可以直接迁移的是：Hub ID、revision、外部仓库 commit、环境版本约束、文件 SHA256（用于复制后校验）、脚本功能分类、历史结果的 evidence class、失败 traceback 和验收标准。

必须在新服务器重新解析或生成的是：

```text
PROJECT_ROOT          = 新服务器上的 guide 工作区
ANOLE_CODE_ROOT       = 新服务器上的 Anole checkout
ANOLE_MODEL_PATH      = 新服务器上的 Anole 权重目录
LUMINA_CODE_ROOT      = 新服务器上的 Lumina checkout
LUMINA_MODEL_PATH     = 新服务器上的 Lumina 权重目录
RUN_ROOT              = 新服务器上的 runs 目录
CLIP_DIR              = 新服务器上的 CLIP 目录（如使用）
PROMPT_MAP            = 新 run 内保存的 prompt map
TEACHERS_JSON         = 新 run 内生成并校验的 teachers.json
```

新服务器第一步应创建一个 `runs/bootstrap/resolved_paths.json`，并让后续命令只引用这些变量或该文件。例如：

```bash
export PROJECT_ROOT=/srv/morph/guide
export RUN_ROOT=/srv/morph/runs
export ANOLE_CODE_ROOT=/srv/morph/external/anole
export ANOLE_MODEL_PATH=/srv/morph/models/anole-7b-v0.1
export LUMINA_CODE_ROOT=/srv/morph/external/Lumina-mGPT
export LUMINA_MODEL_PATH=/srv/morph/models/lumina-mgpt-7b-768-omni
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$RUN_ROOT/bootstrap"
python - <<'PY'
import json, os
p = {k: os.environ[k] for k in (
    "PROJECT_ROOT", "RUN_ROOT", "ANOLE_CODE_ROOT", "ANOLE_MODEL_PATH",
    "LUMINA_CODE_ROOT", "LUMINA_MODEL_PATH", "CUDA_VISIBLE_DEVICES")}
json.dump(p, open(os.path.join(p["RUN_ROOT"], "bootstrap", "resolved_paths.json"), "w"), indent=2)
PY
```

旧命令只能作为参数模板。新服务器必须先运行 `realpath`/`test -e` 检查，替换所有路径，再把最终命令保存到对应 run 的 `command.sh`。旧的 `metrics.jsonl`、prompt map、suffix state 和图片不能因为路径被替换就变成新服务器结果；它们仍是历史参考。

## 1. 代码 provenance、patch 和脚本 hash

### 当前 guide

```text
path: /home/yding/workbar/guide
git rev-parse HEAD: MISSING (fatal: not a git repository)
git status/diff: MISSING (同上)
```

因此不能报告“当前 guide commit”或自动生成未提交 patch。迁移包应附源码 manifest，并在新服务器建立新的初始 commit；不能把外部 Anole commit 冒充 guide commit。

### 外部代码

```text
/home/yding/workbar/morph-runtime/external/anole
  commit: af563fa13ddc9d86503d91e6183a05ba60546c14
  working tree: clean

/home/yding/workbar/morph-runtime/external/Lumina-mGPT
  commit: a28a38fb6de20f7067a03f5879813e3b6d811d18
  working tree: ?? lumina_mgpt/ckpts/
```

当前工作区关键脚本 SHA256：

```text
scripts/06_route_bridge_sweep.py       69ccf73437b4d8f8dd51ca3a19128d5631105873f6ce5b0271245cd1264dcd85
scripts/09_project_anole_soft_suffix.py d1d22263794f5240b6536d294015783e3d9011b68aa30f942a88ab9f8956e1c6
scripts/14_adam_soft_route.py           e0c520a3a02fb81c72ed94d2c9f9ab77a25f7570ee6dbe4ffa697f0cffac3a4d
scripts/28_eval_post_boi_protocol.py   ae4155e1bf45e634353ec24057d4ccc4e9e2a6a3ee15fe8f8e13a3a0f7e60034
scripts/37_anole_apple_route_only.py   abc2ff9b18ee1582d16d89c72738838b465648809e50b8be4685fee4f1104243
scripts/65_diag_hard_gcg.py             6c9d334c28743407935a066dd482acf998bfcbf7a3a30c507c6924f2b5ccc8ac
scripts/P0_backward_feasibility.py     40567fc634840b158f4188c85bf157caa2b4d98b33c5499a784bd547e0f4dd20
scripts/P0_verify_full_forward.py      d325fe00c66bf4cb50917db264563357ad3438c662b757c5b76f18b94f4b5d78
```

新服务器迁移后应重新执行 `sha256sum`，并把结果写入源码 manifest。

## 2. 模型、Tokenizer、VQ 路径与 SHA256

### Anole

```text
Hub: GAIR/Anole-7b-v0.1
revision: b4d4d6d06d02d0b0e95ac321551b79739579130c
code: /home/yding/workbar/morph-runtime/external/anole
code commit: af563fa13ddc9d86503d91e6183a05ba60546c14
model: /home/yding/workbar/morph-runtime/models/anole-7b-v0.1/models/7b/consolidated.pth
text tokenizer: /home/yding/workbar/morph-runtime/models/anole-7b-v0.1/tokenizer/text_tokenizer.json
VQ config: /home/yding/workbar/morph-runtime/models/anole-7b-v0.1/tokenizer/vqgan.yaml
VQ checkpoint: /home/yding/workbar/morph-runtime/models/anole-7b-v0.1/tokenizer/vqgan.ckpt
```

```text
consolidated.pth  34c6d623664c842bf0bc5f652c8ce3fc477052cefad9549dbf777114d3060639
text_tokenizer.json afe03e2b29fde3e375548130dfcf7b296b55bbc960c4994c4ead785af2867b86
vqgan.yaml       2b35c298a9bca9d8f0ab937cca0d63e081b69595dae29116872c58b93e8775ed
vqgan.ckpt       4ede986bf6b171db3081ce171ad88e4ac970793cea14c180b3e5ac5105f4cb43
```

文件大小（bootstrap manifest）：14,026,454,134；5,629,972；1,488；281,270,377 bytes。

### Lumina

```text
Hub: Alpha-VLLM/Lumina-mGPT-7B-768-Omni
revision: c24a700e56cd319917598782ede8b126a4eb1b5d
code: /home/yding/workbar/morph-runtime/external/Lumina-mGPT
code commit: a28a38fb6de20f7067a03f5879813e3b6d811d18
model: /home/yding/workbar/morph-runtime/models/lumina-mgpt-7b-768-omni
tokenizer source: Alpha-VLLM/Lumina-mGPT-7B-768, revision 755e7e40530a8edf4eea6078cbc484d2bfda091f
VQ source: Meta Chameleon official licensed release
```

```text
model-00001-of-00002.safetensors 64555dead2ff6f2b67eec922181c9b46f95652c4af3cb653130d613de1f1f2c8
model-00002-of-00002.safetensors 3b49ec4dcb7fdebca2bb49916ca6837ad759ccfeebb84e9abdcabfad512f81d8
tokenizer.json                   b7b56bbd14f47aeab7561966629ed22d5d027441c99b05240a6120f97ee8dab3
```

Lumina bootstrap manifest 记录了成功 load/forward（logits `[1,6,65536]`），但 VQ decode 为失败：`Meta Chameleon VQ files not installed yet`。因此 Lumina 当前是 gateway/load evidence，不是完整图像攻击 evidence。

## 3. 精确环境和硬件

### Anole freeze

```text
Python 3.10.x
torch 2.4.0
torchvision 0.19.0+cu121
torchaudio 2.4.0+cu121
CUDA build 12.1
xformers 0.0.27.post2
triton 3.0.0
numpy 1.26.4
tokenizers 0.19.1
custom transformers: GAIR-NLP/anole, af563fa13ddc9d86503d91e6183a05ba60546c14
```

### Lumina freeze/bootstrap

```text
Python 3.10.20
torch 2.3.0+cu121
torchvision 0.18.0+cu121
torchaudio 2.3.0+cu121
CUDA build 12.1
transformers 4.43.3
triton 2.3.0
xllmx: a28a38fb6de20f7067a03f5879813e3b6d811d18
GPU in bootstrap manifest: NVIDIA L40
```

历史机器文档写过 8×L40；具体 run 使用哪一张卡必须以该 run 的 `command.sh`/`process.txt` 为准。Anole E0 audit 的模型参数 dtype 是 `torch.bfloat16`。未找到每个历史 run 都保存 `dtype/backend/GPU model` 的完整记录，故这些字段对历史 run 不是全覆盖 PASS。

当前 shell 不能代表旧环境：Python 3.9.25，torch import 失败，`nvidia-smi` 无法连接 driver。

## 4. 从初始化到输出的实际命令

环境脚本位于：

```text
scripts/bootstrap_anole_env.sh
scripts/bootstrap_lumina_env.sh
```

Anole 脚本实际安装命令是：

```bash
conda/micromamba create -n morph-anole -c conda-forge python=3.10 pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
python -m pip install xformers==0.0.27.post2
python -m pip install huggingface_hub==0.23.4 tokenizers==0.19.1 regex \
  python-dotenv==1.0.1 numpy==1.26.4 Pillow==10.0.1 packaging pyyaml tqdm requests safetensors
python -m pip install -e /path/to/external/anole/transformers --no-deps
python -m pip freeze > environment/anole.freeze.txt
```

实际 bridge formal command（旧绝对路径，仅作 provenance；新服务器必须替换路径）：

```bash
/home/yding/workbar/morph-runtime/bin/micromamba run -n morph-anole \
python /home/yding/workbar/guide/scripts/06_route_bridge_sweep.py \
  --model-path /home/yding/workbar/morph-runtime/models/anole-7b-v0.1 \
  --code-root /home/yding/workbar/morph-runtime/external/anole \
  --output-dir /home/yding/workbar/morph-runtime/runs/03_anole_bridge_route_sweep/results \
  --alphas 1.0 0.75 0.5 0.25 0.0 \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
           16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
  --max-steps 128 \
  --prompt-ids red_apple golden_retriever yellow_school_bus blue_ceramic_cup
```

真实 hard-route 链是：`script 37`（run 68 → 69 → 72）→ `script 09` → `script 65 --objective boi-ce` → script 10 native evaluation。运行配置和 init 路径见 `runs/68_quant_mstep_64_l01/results/config.resolved.json`、`runs/69_quant_mstep_64_l20/results/config.resolved.json`、`runs/72_quant_mstep_64_l50/results/config.resolved.json`、`runs/101_gcg_boice_run72/gcg_projection.json`。

旧 run 101/103 的 evaluator 命令还引用了 `/home/yding/.claude/jobs/.../prompt_map_*.json`，这些临时文件不能迁移为依赖。

## 5. 最小 baseline、parity、backward、finite difference、cache

| 检查 | 当前证据 | 状态 |
|---|---|---|
| native ordinary-text baseline | `runs/01_anole_baseline_matrix/metrics.jsonl`，student-only 无 BOI、无 visual token | `PASS`（旧环境） |
| teacher image control | 同一文件 teacher-only，forced BOI，1024 visual tokens | `PASS`，但仅 control |
| official forward parity | `P0_verify_full_forward.py` 仍硬编码旧路径且无可靠 CLI | `MISSING` 作为迁移证据 |
| E0 branch cache vs uncached | `abs_diff=0.0`，content loss -1.968749 两路相同 | `PASS`，仅该分支/长度 |
| suffix backward finite | E0 gradient finite/nonzero，但 finite difference rel error 0.81–1.27，多处 sign 不同 | `FAIL` 数值验收 |
| descent check | boi-ce alpha .01/.03、joint .01/.03/.1 未下降；仅部分步长下降 | `FAIL/PARTIAL` |
| cache continuation vs full-prefix | 未找到保存的 continuation error、position sweep 或 full-prefix 数值 artifact | `MISSING` |

E0 原始记录：`/home/yding/workbar/morph-runtime/runs/106_e0_joint_audit_k1/results/audit.json` 与 `metrics.jsonl`。其中明确记录：`model_param_dtype=torch.bfloat16`、gradient shape `[64,4096]`、`boi_ce_gradient_norm=49.5`、`content_gradient_norm=56.0`、`joint_gradient_norm=93.5`，但这只能证明图有 finite tensor，不能证明梯度方向/数值正确。

## 6. xformers backward 的真实结论

真实失败日志：

```text
/home/yding/workbar/morph-runtime/runs/04_anole_soft_route_distill_128_retry7/stdout.log
...
File .../xformers/ops/rope_padded.py, line 93, in rope_padded
    raise ValueError("Gradients not supported.")
ValueError: Gradients not supported.
```

这不是 flash kernel registration warning；它是 suffix M-step 实际触发的 backward failure。日志中的 `xformers::flash_fwd` / `flash_bwd` FutureWarning 不能作为 backward 成功证据。

后续 custom analytic 路径位于 `scripts/P0_backward_feasibility.py`，使用官方 `rope_padded` forward、analytic inverse RoPE、xformers attention backward 和 RMSNorm backward。当前没有足够 artifact 证明它在新环境通过了完整 finite-difference/parity。因此：

```text
stock xformers rope_padded backward: FAIL
custom analytic backward: NOT YET VERIFIED
custom wheel/source patch: 未发现可追溯 patch/wheel，MISSING
eager/SDPA generation smoke: 可作为 fallback，但不等价于 analytic M-step
```

## 7. cache/past_key_values 类型和 position

Anole 不使用 Hugging Face `past_key_values` API。实际类型是：

```python
LayerCache = tuple[torch.Tensor, torch.Tensor]  # (K, V), per layer
cache = transformer.make_cache(model.args, length=max_length, device="cuda", dtype=...)
```

`Branch`（scripts 12/13）先做 full-prefix prefill，再 one-token advance；bias 使用
`q_seqlen=[len(token_ids)]`、`kv_seqlen=[len(token_ids)]`、`kv_padding=max_length`；每次 advance 后执行 `bias.k_seqinfo.seqlen.add_(1)`。position 由当前 cache sequence length 管理，而不是 HF `cache_position` 参数。

未找到真实的“cache continuation vs full-prefix”误差、长序列 position sweep 或多层 cache shape artifact。新服务器必须新增这一项，至少保存每个 length 的 max abs/logit error、首 visual token position、KV length before/after。

## 8. scripts 06/09/14/28/37/65 的来源与证据类别

这些脚本来自当前 guide 工作区，当前 guide 无 Git commit；上文给出的 SHA256 是唯一可复核的源码版本标识。功能和证据分类如下：

| script | 真实作用 | teacher / forced BOI / visual prefix | 证据类别 |
|---|---|---|---|
| 06 | teacher/student synchronized local bridge；common warper；`alpha` 插值 | teacher logits 在 bridge 中使用；不 forced BOI；无 visual prefix | bridge/oracle |
| 09 | continuous suffix 最近邻投影到 ordinary text IDs | 不使用 teacher；无 forced BOI；无 visual prefix | projection diagnostic |
| 14 | 16-slot exact-forward Adam soft route M-step | bridge elites 作为训练 target；训练期 teacher/bridge；部署评估必须 student-only | soft/control |
| 28 | faithful post-BOI soft evaluator | student-only 可 native-style；采样 BOI 后回填再采 visual token；不应强制 BOI | evaluator/protocol |
| 37 | teacher-text soft route、quantization-aware M-step；run 68/69/72 主链 | `--teacher-text` 只在训练构造 target；软 suffix 是 continuous embedding；无部署 visual prefix | soft/control；量化链用于 hard init |
| 65 | hard token GCG/BOI-CE 诊断；可选 joint content objective | 搜索阶段可用 teacher visual targets；无部署 visual prefix；结果必须 script 10 native readback | search diagnostic；native status 由独立 evaluator 决定 |

## 9. 历史结果、原始 trace、config、seed、suffix 与 evidence class

### 可直接追溯的结果

| run | config/seed/suffix | 原始证据 | 类别 |
|---|---|---|---|
| 01 baseline | `metrics.jsonl`；teacher seeds 0/1，student seeds 0/1；无 suffix 或 teacher forced route | JSONL 包含 `forced_boi_positions`、`visual_token_count`、`structure_valid` | teacher control + native baseline |
| 03 bridge sweep | 4 prompts × 32 seeds × alphas 1/.75/.5/.25/0；suffix 空字符串 | `results/metrics.jsonl`，字段含 alpha、teacher/student prompt、support overlap、BOI | bridge/oracle |
| 37 | 16 slots、40 steps、lr .05、init seed 0 | `runs/37_apple_route_only/results/config.resolved.json`、metrics、soft_suffix.pt | soft route/control |
| 51 | Asian teacher text × sushi；同上 | config + metrics + soft state | soft teacher-text control |
| 52 | Asian teacher text × kimchi；同上 | config + metrics + soft state | soft teacher-text control |
| 53 | African teacher text × ethiopia_coffee；同上 | config + metrics + soft state | soft teacher-text control |
| 68→69→72 | 64 slots；quant penalty .1→2→5；run 69/72 init 前 run suffix | 每个 run 有 config、metrics、soft_suffix.pt、selected_elites.json | soft quantization/projection diagnostic |
| 73 | run-72 projection；seeds 32–63；prompt map 在临时 job | config、metrics、readback | hard native evaluation，但 provenance 不可直接迁移 |
| 101/103 | run-72 projection → BOI-CE GCG；101 seeds 0–7，103 seeds 8–71 | `gcg_projection.json`、各自 config/metrics/readback | hard native evaluation；合并 reported 41/80 |
| 106 E0 | suffix length 64、strawberry、content step 1 | audit + metrics，含 cache equality/finite difference/descent | algorithm audit；不是成功 route/content |

### suffix 字符串

run 37/51/52/53 的 suffix 是连续 embedding，`soft_suffix.pt` 中没有普通文本 suffix string；应记录 `suffix_string: N/A; suffix_type: continuous_embedding; suffix_length: 16`。run 101/103 的 ordinary hard suffix token IDs 在 `gcg_projection.json`；若要报告可读字符串，必须在新环境用同一 tokenizer decode 并保存结果。旧 run 中没有统一保存 decoded suffix string，属于 `MISSING`。

### 原始 trace 缺口

部分 run 只有 metrics/config/readback，没有完整 stdout/stderr、环境打印和 traceback；尤其 37/51/52/53/68/69/72/101/103 的完整启动日志并未全部保存。不能把 config 当作原始 trace。迁移重跑时每个命令必须用 `tee stdout.log`，保存 exit code、环境摘要和 resolved config。

## 10. native 与诊断结果的严格分类

### Native T1

必须同时满足：冻结模型；普通文本 prompt；ordinary text token suffix；`alpha=0`；无 teacher logits/KV；无 forced BOI；无 visual prefix/state injection；官方 faithful decoder。run 101/103 的 hard projected/GCG suffix 在 script 10 的 student-only native evaluation 中属于该类，前提是新服务器重新生成同等 artifact。

### 诊断/control

teacher-only、forced BOI、bridge `0<alpha<1`、continuous soft suffix、teacher text target、teacher visual distribution、state injection、mode flags、output prefill、forced image route、GCG loss/teacher-forced content margin 均属于诊断或 control。它们可以说明机制或上界，不能进入 native success rate。

## 11. 新服务器必须补齐的最小证据包

```text
[ ] 新服务器 git initial commit + source manifest + explicit patch/provenance
[ ] preflight: Python/Torch/CUDA/GPU/dtype/backend
[ ] model/tokenizer/VQ manifest + SHA256
[ ] native baseline + teacher control
[ ] official vs differentiable forward parity（短序列和 route 长度）
[ ] suffix backward：gradient finite、finite difference、descent
[ ] cache continuation vs full-prefix，含 position/cache length 参数
[ ] stock xformers failure 或 custom analytic success 的完整 stdout/stderr
[ ] scripts 06/09/14/28/37/65 的源码 hash 和 resolved config
[ ] 101/103 同一 hard suffix 的 decoded token string、prompt map、seed split
[ ] 每个 substantial run 的 stdout.log、stderr.log、exit_code、config、metrics
[ ] native / soft / teacher / bridge / forced 结果分栏
```

在这些 artifact 完成前，准确的表述是“迁移规格和历史 provenance 已整理，核心梯度/cache 复现尚未完成”，而不是“实验框架已经复现”。
