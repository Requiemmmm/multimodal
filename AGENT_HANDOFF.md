# 给新服务器 Codex 的第一条指令

你在新服务器上工作。先执行：

```bash
git rev-parse HEAD
git status --short
sed -n '1,260p' NEW_SERVER_REPRODUCTION.md
sed -n '1,260p' docs/reproduction_2026-10-15/MORPH_PERSON_CONTENT_REPRO_REPORT.zh-CN.md
```

不要假设旧服务器的 `/home/yding/workbar/morph-runtime` 路径存在。把所有路径改为当前服务器的 `MORPH_ROOT`。报告中的 `environment.json`、`model_manifest.json`、各组 `config.resolved.json`、`command.sh`、`metrics.jsonl`、`selected_elites.json` 和 PNG 是 reference evidence；模型权重不在仓库内，需要重新下载并核验 SHA256。

复现顺序：环境/模型审计 → teacher target token 对照 → run 51 40-step suffix → 1 个 seed faithful post-BOI 对照 → 全部 16 seed → run 52/53。每一步保存 command、resolved config、stdout/stderr、exit code，并报告首次差异层级。必须把 route、BOI/modality switch、valid image、person content、object content 分开。
