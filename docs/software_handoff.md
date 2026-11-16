# 四部分核心软件交接

更新于 2026-11-16。本页对应本地 `codex/phase3-goal-workflow` 工作区，起始提交为 `e3a6ef891f47a1cfd779eb710ed9bf0fd46bd398`。当前新增代码尚未提交或推送；远端 `codex/s2-single-arm` 和服务器已有的 main 不会自动包含这些代码。发布后必须在服务器记录实际新提交号，不能用本页的起始提交冒充完成版本。

I-01 整体验收已完成：根 agent 全量 **353 项测试通过（245.475 秒）**，一名实现与两名独立审核完成全部代码卡；工程最终另跑 66 项针对性回归，研究另跑 7 项整体验收和配对基线探针，均通过。状态见 [任务卡](task_cards.md)，逐文件记录见 [CHANGELOG](../CHANGELOG.md)。旧已发布版本的安装与第一二部分 pilot 操作保留于 [历史服务器交接](server_handoff.md)。本轮先完成软件，未启动真实模型、真实容器或正式数据采集。

## 四部分的入口

| 部分 | 可运行的软件步骤 | 主要代码 |
| --- | --- | --- |
| 1 | 来源历史 → 教师候选 → 恢复审核/独立 K=2 回放 → 冻结记忆库 | `memory_generation/` |
| 2 | 缓存原动作和公开特征 → 可恢复配对或同一活会话随机单臂 → 只读审核数据集 | `target_data/`、`interventions/` |
| 3 | 收益/风险/成本拟合 → 独立校准 → 监督 gate → 配对或 DR 约束 bandit | `policy/dataset.py` 至 `policy/bandit.py` |
| 4 | 冻结包 → 新完整 validation rollout → 配对或保守选择锁 → 分组分配的最终 test → 报告 | `policy/frozen.py`、`rollout.py`、两种 selection、`final_*.py` |

完整命令与参数见 [训练指南](policy.md)、[评估指南](evaluation.md) 及 `cave3 <command> --help`。每次运行使用新的输出路径；保留失败、pending 和旧结果，不覆盖或自动重试。

## 在服务器验证软件

先把完成版本发布或完整传输到服务器的独立目录，再运行以下命令。不要在正在采集的工作区直接切换代码，也不要把旧分支 clone 成功当作拿到了本轮实现。

```bash
git branch --show-current
git rev-parse HEAD
git status --short
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[policy]'
.venv/bin/python scripts/run_tests.py
.venv/bin/python -m cave_mem_v3 --help
.venv/bin/python -m cave_mem_v3 check --config configs/phase1/server.toml
```

需要 Python 3.11+；本机验证环境为 Python 3.14.8 / NumPy 2.5.4。`policy` 是新增可选依赖，基础采集不依赖 NumPy；真实检索另需 `.[encoder]` 和本地已固定版本的 encoder，CLI 不替你下载权重。标准 editable 安装已在本机通过。现有 `server.toml` 的只读检查返回 `ready: true`、`model_downloaded: false`；ready 仅说明必填值存在，不核验服务、模型身份、容器或资产，也不是正式运行许可。

`runs/`、`data/`、权重、密钥和 `*.local.toml` 不在 Git 中。训练和选型的数据加载器会重读原始轨迹、来源和任务资产，传输派生 dataset/model JSON 并不足够；不要手工重封 hash 或改标角色来迁移。最终冻结计划则保存自包含策略及选择锁快照，执行不需要历史结局，但仍需对应配置、任务资产、环境和模型服务。

完整离线演示从仓库根目录执行：

```bash
.venv/bin/python examples/four_phase_learning_demo.py --output runs/four-phase-learning-demo
```

该目录必须尚不存在。脚本从原始来源回放及训练/校准日志实际拟合模型和策略，运行新验证选型与最终 test；`summary.json` 汇总拟合/选择/最终完整性，`selection.lock`、`final.plan`、`final.report` 可独立重核，原始证据保留。它不导入测试文件、不手填学习权重、不重封结果标签。所有设置和脚本 actor/encoder 都明确为 fixture，不是服务器正式参数。

本机最终验收目录为 `runs/i-01-fixture-final-20261116/`（不随 Git 上传）。实际输出：7 个预测头和初始 gate 均 fitted，配对/单臂 bandit 均生成候选，单臂 DR 的 24 行/24 组有完整记录；新验证选中 paired-bandit，最终 12/12 个任务有结局。学习策略分配到 7 任务并实际提醒 3 次，no_memory 分配到 5 任务；原始来源、配对和单臂证据 hash 均不变。这些数值仅证明软件路径被执行，不证明研究有效性。

## 实际运行前必须确定的研究配置

冻结完整 manifest 的 history/target_train/calibration/selection/test 分区与 task/template/dependency groups；声明来源和机制是否真正留出。固定 benchmark revision、资产/镜像、actor/teacher/encoder 版本、提示、工具权限、预算、种子、效用权重、阈值和停止规则。真实服务必须返回可审计且一致的 model/fingerprint；缺少身份时选型被阻断，不能补造指纹。

先完成来源协议兼容性和实际恢复审核，再进行来源 pilot。超出已有 Terminal 文件系统协议的任务先提出具体 adapter 或协议改动；不能用本次 fixture 的成功替代恢复实测。不可恢复目标显式选择单臂路线和保守选型，不能在配对恢复失败后悄悄降级。

两组 K=2 修正全部成功、两组原始/修正回放完整有效且平均效用差严格超过 epsilon，才入新主库。这里的来源有效不代表目标提醒必然有益；后续训练和验证仍必需。单臂只有观察到的一臂，不含已观测个体 H/rescue。

## 当前实现的统计范围

- `x` 使用冻结的上下文/原动作向量与显式结构特征；没有新增一套对记忆全文的联合编码。实际记忆证据、适用性、暴露、差异和检索信号有明确数值/未知语义。
- 收益/成本校准是组级实现值预测范围，风险是分箱组级上界；它们不是逐点条件均值置信区间或安全定理。
- bandit 的预算单位是 `selected_gate_actor_calls`：选中草稿调用加干预增量调用。完整任务 token、工具、时间和费用另报，不能宣称已受这个预算约束。
- 单臂 DR 与配对直接目标分开。交叉拟合、ESS 和截断诊断不能消除未知结局或缺支持问题；单臂风险约束采用明确最坏情况界。
- 单臂 DR 统计可以独立使用；当前单臂 bandit/freeze 仍需要先有兼容的配对 model/calibration/warm_gate。没有任何配对数据的纯单臂从零初始化/校准尚不支持，不能把本示例解释成已解决这一问题。
- 选择依据新完整 validation 任务效用，同时用独立配对审核或保守 live 界检查风险/调用预算。最终 test 按独立组分配策略；报告全任务分母、HT 边际估计与联合组 bootstrap。验证风险不是已测得的 test 个体回退风险。
- 共用一个来源库不等于来源留出泛化。自然样本、四格/噪声库、第二档真实 actor、外部基线原系统复现与可选 Inspect 尚需单独的研究任务。

## 下一张真实运行任务：R-01 来源小样本闭环

- **Goal：** 在服务器冻结一批真实 history 任务和运行配置，完成历史采集、教师候选、独立恢复审核、K=2 回放；有合格项时冻结新库。
- **Context：** 核心软件有离线检查，尚无本轮真实训练和最终评估数据；服务器需要先拿到完成版本并复核软件，旧临时库和模板不能直接作为正式输入。
- **Rules：** 只访问 history；保留全分母、失败和 pending；不看 test、不以零入库为由改阈值/重采、不跳过恢复/身份门禁。先冻结模型/任务/资产/预算，超出协议先审阅适配。
- **Acceptance：** 每个计划运行、候选和两组回放都有可追溯记录；来源证据可按新规则独立重核，合格数可以为零；报告实际提交、配置、模型、成本、排除及未解决限制。完成后再定义一张目标数据采集卡。

这张真实运行卡只作交接，不由本次软件目标自动启动。以后代码变更继续采用一名 writer、两名独立 reviewer；一次实施一张卡，审核和所需测试通过后再推进下一张。
