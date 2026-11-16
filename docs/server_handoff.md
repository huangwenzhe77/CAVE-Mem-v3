# CAVE-Mem-v3 服务器交接

> 2026-11-16 本地开发说明：下文是已发布 `codex/s2-single-arm` 的历史交接快照。四部分新软件交接见 [software_handoff.md](software_handoff.md)，对应本地 `codex/phase3-goal-workflow`，尚未提交/推送，不能假设远端已有这些改动。当前进度以 `docs/task_cards.md` 和 `CHANGELOG.md` 为准；新的开发协作为一名实现、两名独立审核。真实采集不在本次软件目标中自动启动。下文旧测试数量、实现范围和协作人数仅记录旧版本。

更新于 2026-11-15。本文件对应 GitHub 分支 `codex/s2-single-arm`。克隆后运行
`git rev-parse HEAD` 固定实际提交；不要仅凭分支名标识实验版本。

## 当前状态

- 第一部分历史采集、教师候选、来源 K=2 配对回放与冻结库的软件流程已实现。新 v6 证据要求两组修正均成功、配对均完整有效，且平均效用差严格大于 `acceptance_epsilon`；旧 v5 证据按旧语义读取，不能冒充新规则的目标实验来源。
- 第二部分已实现目标检查点抽样、检索与 `a0/x_t` 冻结，以及可恢复检查点的 Silent/Remind 配对采集和不可恢复检查点的在线随机单臂采集。单臂保存已知正分配概率，只观察被分配的一臂，不生成配对 `D/H/rescue`。
- 本地 `python scripts/run_tests.py` 通过 220 项离线软件测试；`runs/s2-02-fixture-20261115/` 是脚本 fixture 示例。它们不是真实模型或 benchmark 的结果。
- 旧 H20 `r3` pilot 的来源库是临时工程产物，仍在未上传的 `runs/` 中；它不是本批新规则的正式来源库。当前没有已验证的新批真实目标提醒收益。第三部分训练、第四部分部署评估及 Inspect 均未实现。

## 克隆与本地检查

以下命令在 Linux 服务器执行。这个分支包含当前代码，GitHub 默认 `main` 分支尚未同步这些改动。

```bash
git clone --branch codex/s2-single-arm --single-branch https://github.com/huangwenzhe77/CAVE-Mem-v3.git
cd CAVE-Mem-v3
git rev-parse HEAD
git status --short
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/run_tests.py
.venv/bin/python -m cave_mem_v3 check --config configs/phase1/server.toml
```

`check` 不会请求模型。克隆后 `data/`、`runs/`、模型权重、API 密钥和本地配置均不存在；它们按 `.gitignore` 留在各自机器上。新机器的 Python 需为 3.11+，真实 Terminal 任务还需可用的 Docker 或 Enroot、任务镜像和 benchmark 资产。目标检索需要已有的本地 encoder 权重，可在需要时安装 `.[encoder]` 可选依赖；CLI 不下载 encoder。

`configs/phase1/server.toml` 记录过旧服务器的模型路径、镜像快照目录和工程预算，`configs/benchmarks/terminal.example.json` 仍含占位任务。它们是参考模板，不能克隆后直接启动真实采集。在服务器复制为被 Git 忽略的 `configs/phase1/server.local.toml`，任务 catalog 放在被忽略的 `data/` 下；填写并冻结实际地址、模型/权重版本、提示、预算、种子、任务资产及依赖组。密钥只通过配置的 `api_key_env` 对应环境变量提供，不写入 TOML、Git 或输出日志。若继续使用 `scripts/phase1_actor_service.py`，先注意该脚本固定读取 `server.toml`，并非上述 `.local.toml`；本地配置下应使用 CLI 或先专门修改该调度脚本。

## 下一张建议任务卡：R-01 真实来源小样本闭环

- **Goal：** 冻结一批真实 history 任务，采集完整轨迹，生成教师修正候选，完成恢复审核和 K=2 来源回放；有合格项时冻结新库。
- **Context：** 代码路线已通过离线测试，但旧 pilot 和模板配置不能作为新批正式实验输入。第二部分的真实目标实验需要新规则合格来源库。
- **Rules：** 只访问 history 任务；按任务、模板和已知依赖组隔离其余角色。先冻结 benchmark commit、资产/镜像哈希、actor 与 teacher 身份、提示、预算、种子和接受阈值。教师候选只是待检验假说；恢复/评分错误或 pending 不得跳过、补跑或重标。超出已审核 Terminal 文件系统协议的任务先提出具体 adapter/协议变更供审阅。
- **Acceptance：** 每个计划运行、候选与两组回放都有可追踪的完成、失败或 pending 记录；验证器能按新 v6 规则独立重核。合格记忆数可以为零，需报告分母、原因、实际模型/成本和未解决限制，不以零产出为由放宽门槛。

先完成 R-01，再选择一张真实 `target_train` 采集卡。不要从 fixture 示例或旧临时库启动第三部分训练。

服务器侧建议按以下顺序推进，每步用新的 `runs/<run-id>/` 目录保存原始产物：

1. 准备本地 benchmark catalog、任务资产、模型服务、镜像及本地配置；运行 `cave3 check --config configs/phase1/server.local.toml`，核对实际模型与资产版本。
2. 用 `cave3 manifest --catalog data/tasks.local.json --output runs/<run-id>/manifest-preflight.json` 固定任务身份；用 `cave3 audit --config configs/phase1/server.local.toml --manifest runs/<run-id>/manifest-preflight.json --task <history-task-id> --output runs/<run-id>/audit-<task-id>.json` 做来源检查点恢复诊断。诊断本身是未批准产物，还需按任务范围审阅并在 catalog 中登记审核记录，然后生成最终 manifest。
3. 只有协议和配置审核通过后才运行 `cave3 collect --config configs/phase1/server.local.toml --manifest runs/<run-id>/manifest-final.json --output runs/<run-id>/history`；再运行 `cave3 candidates --collection runs/<run-id>/history --generate --config configs/phase1/server.local.toml --output runs/<run-id>/candidates.json`，检查并冻结候选。
4. 对完整候选运行 `cave3 verify --config configs/phase1/server.local.toml --manifest runs/<run-id>/manifest-final.json --collection runs/<run-id>/history --candidates runs/<run-id>/candidates.json --output runs/<run-id>/verification`。通过项会写入 `verification/bank.json`；零候选或零入库均报告实际原因，不自动换任务或阈值。

命令中的 `<run-id>` 和 `<history-task-id>` 必须换成预先选定的值；这是一份执行顺序，不是克隆后自动运行的脚本。详细任务协议、恢复审核与每条命令的条件见 [第一部分指南](phase1.md)；目标阶段命令及两条采集路径见 [README](../README.md)，研究边界见 [实验设计](experiment_design.md)。

## 产物与交接边界

`runs/`、`data/`、`.env`、`*.local.toml`、`models/`、`weights/` 被 Git 忽略，克隆不会带来旧实验数据或凭证。保留失败和未完成产物，不在原路径覆盖；请求完成状态不明时先核实，再决定新 run。每次报告实际 Git 提交、配置/资产哈希、计划与完成数量、失败和排除、模型身份、预算与成本；把软件测试和真实 benchmark 观察分开。

后续代码任务继续遵守 [AGENTS.md](../AGENTS.md)：每轮只执行一张明确任务卡，由两名实现 agent 编写，一名独立 agent 审核，完成后逐文件报告并停止。
