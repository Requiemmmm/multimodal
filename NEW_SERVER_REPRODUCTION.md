# 新服务器复现入口

本仓库只包含 MORPH 源码、配置、环境约束和复现文档，不包含模型权重、旧运行目录或临时 prompt map。

## 1. 克隆与路径

```bash
git clone https://github.com/Requiemmmm/multimodal.git
cd multimodal
export MORPH_ROOT=/srv/morph
export PROJECT_ROOT=$PWD
export RUN_ROOT=$MORPH_ROOT/runs
export EXTERNAL_ROOT=$MORPH_ROOT/external
export MODEL_ROOT=$MORPH_ROOT/models
export HF_HOME=$MORPH_ROOT/hf
export ANOLE_CODE_ROOT=$EXTERNAL_ROOT/anole
export ANOLE_MODEL_PATH=$MODEL_ROOT/anole-7b-v0.1
export LUMINA_CODE_ROOT=$EXTERNAL_ROOT/Lumina-mGPT
export LUMINA_MODEL_PATH=$MODEL_ROOT/lumina-mgpt-7b-768-omni
export CLIP_DIR=$MODEL_ROOT/clip-vit-base-patch32
mkdir -p "$RUN_ROOT/bootstrap" "$EXTERNAL_ROOT" "$MODEL_ROOT" "$HF_HOME"
```

新服务器不要假设 `/srv/morph`；只需把 `MORPH_ROOT` 改为高速盘。所有实验命令都应使用这些变量，或先写入 `runs/bootstrap/resolved_paths.json`。

## 2. 环境和代码

```bash
bash scripts/bootstrap_anole_env.sh
# Lumina 需要时再执行：
# bash scripts/bootstrap_lumina_env.sh

git clone --branch <commit> https://github.com/GAIR-NLP/anole.git "$ANOLE_CODE_ROOT"
git -C "$ANOLE_CODE_ROOT" checkout af563fa13ddc9d86503d91e6183a05ba60546c14
```

Anole 起点：Python 3.10、Torch 2.4.0/cu121、xformers 0.0.27.post2、triton 3.0.0。Lumina 起点：Torch 2.3.0/cu121、transformers 4.43.3、triton 2.3.0、Lumina commit `a28a38fb6de20f7067a03f5879813e3b6d811d18`。实际 GPU、CUDA driver、dtype 和 backend 必须写入 bootstrap artifact。

## 3. 模型资产

```bash
MORPH_ROOT=$MORPH_ROOT bash scripts/download_models.sh anole
MORPH_ROOT=$MORPH_ROOT bash scripts/download_models.sh lumina
```

下载后先运行 `scripts/00_audit_anole_assets.py`、`scripts/02_audit_anole_support.py` 和 `scripts/01_smoke_anole_vq.py`。模型 revision、文件大小和 SHA256 写入 `runs/bootstrap/model_manifest.json`；不要把旧服务器绝对路径复制到 manifest。

## 4. 最小复现顺序

1. `scripts/05_anole_baseline_matrix.py`：ordinary text native baseline、authorized image teacher control、empty suffix control。
2. `scripts/06_route_bridge_sweep.py`：bridge 只作 oracle/control，不能计入 native rate。
3. `scripts/14_adam_soft_route.py` 或 `scripts/37_anole_apple_route_only.py`：continuous soft suffix；训练期 teacher/bridge 与 student-only evaluation 分开。
4. `scripts/28_eval_post_boi_protocol.py`：采样 BOI 后回填，再采样 visual token。
5. `scripts/37` 的 64-slot quantization-aware chain：penalty `0.1 → 2.0 → 5.0`。
6. `scripts/09_project_anole_soft_suffix.py` → `scripts/65_diag_hard_gcg.py --objective boi-ce` → `scripts/10_eval_anole_hard_suffix.py`。
7. 只有最后一个 evaluator 的 ordinary text suffix、`alpha=0`、无 teacher/forced BOI/visual prefix 结果可以进入 native 表。

## 5. 必须补齐的验证

在报告任何复现成功前，保存：

- official/differentiable forward parity；
- suffix backward、finite-difference、descent test；
- cache continuation 与 full-prefix 对照；
- `past_key_values` 类型说明（Anole 实际是每层 `(K,V)` tuple）；
- xformers stock backward traceback 或 custom analytic backward 的完整通过证据；
- 每个 run 的 command、resolved config、stdout/stderr、exit code、metrics、seed 和 prompt map。

`docs/MORPH_MIGRATION_EVIDENCE_AUDIT.md` 是旧服务器 evidence ledger，不是可直接执行脚本。旧路径只表示 provenance；新服务器必须重新解析路径并重新生成结果。
