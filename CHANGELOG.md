# 变更记录

## 2026-11-15 — S2-02 不可配对目标的随机单臂采集

在 `ee97ac12d4e43d4f168e6ce2acb5f2196c1a56f5` 基础上使用分支 `codex/s2-single-arm`；两名实现 agent、一名独立审核 agent 协作，主 agent 集成验证。保留开始前所有未提交改动；未提交或推送。本条只列 S2-02 相对开始时工作区的新增/修改。

| 文件 | 本轮改动 |
| --- | --- |
| `src/cave_mem_v3/interventions/single.py` | 新增在线前缀采集，在活会话中先冻结 a0/x 再随机分配一臂；记录严格正概率与独立种子、写前事件链、排除/失败/pending；冻结身份漂移及清理错误停止整批，完成后重入不构建工厂或调用模型 |
| `src/cave_mem_v3/interventions/execution.py` | 新增 `run_live_target_arm`，复用 Silent 缓存动作、Remind 单次提醒与相同普通续跑预算；活会话不调用快照、恢复或状态指纹，原配对路径的恢复要求保留 |
| `src/cave_mem_v3/interventions/single_outcomes.py` | 新增只读原始日志校验和观测结果汇总；重核输入/分配/成本/终局、跨任务模型身份及停止顺序；费用未知不生成效用，单臂不生成 D/H/rescue |
| `src/cave_mem_v3/cli.py` | 新增 `collect-single-target` 与 `summarize-target-single`；显式要求分配概率，仅接收 target_train，pending 返回非零 |
| `examples/phase2_single_demo.py` | 新增离线来源验证到在线单臂的完整示例；环境不暴露 snapshot/restore/state_fingerprint，已有产物不能覆盖 |
| `tests/test_single_execution.py` | 新增 16 项执行与编排测试，覆盖两种臂、缓存动作、单次提醒、不可恢复环境、采样失败、未知完成、跨任务配置/身份漂移、零工厂重入 |
| `tests/test_target_single_cli.py` | 新增 15 项 CLI/汇总测试，覆盖正概率、分区、未知费用、排除分母、重封后不一致记录、共享错误停止边界及离线示例 |
| `AGENTS.md` | 持久化用户要求：后续代码任务由两名实现 agent 和一名独立审核 agent 协作，分配文件所有权 |
| `README.md` | 更新第二部分两条采集路径、单臂命令、概率含义、未知结果和适配边界 |
| `docs/experiment_design.md` | 同步配对/单臂软件状态，区分单臂分配概率与配对执行顺序概率，明确统计估计和训练尚未实现 |
| `docs/task_cards.md` | 新增并完成 S2-02 的 Goal/Context/Rules/Acceptance 与执行记录；保留旧细分卡为历史参考 |
| `CHANGELOG.md` | 本条逐文件交付、验证、实际输出和限制 |

验证：最终 31 项单臂针对性测试通过；`python scripts/run_tests.py` 共 220 项软件测试通过，`git diff --check` 通过；独立审核无剩余阻断问题。审核原始记录保存在本机临时目录 `C:/Users/260917-1/AppData/Local/Temp/cave-mem-v3-s2-02/review.txt`。本轮没有修改来源入库或配对结果定义。

运行 `python examples/phase2_single_demo.py --output runs/s2-02-fixture-20261115`：fixture actor/encoder，seed=42，最多 4 步/4 次模型调用，来源 K=2，检查点 hazard=0.99，`P(Remind)=0.5`，模型调用成本权重 0.01。计划并完成 1 个 target rollout，仅分配 Remind，实际观察成功、效用 0.96；共享准备与实际分支合计 4 次脚本 actor 调用、3 次工具动作、0 次外部模型请求。准备消耗 1 次调用，分支最多使用 1 次提醒重提及 2 次普通续跑。输出包含分配概率 0.5 和单独的检查点纳入概率 0.99。主 agent 重新只读汇总与保存摘要完全一致，并核对没有任何文件被改写。

限制：以上均为离线软件测试/示例，不证明提醒的真实收益或个体因果效果。当前仅接收 target_train 与同一完整 manifest 的 history 来源新 v6、K=2 库；真实 Terminal 仍限于已审核的文件系统协议，不自动支持任意不可恢复后端。无记忆/预算不足是不随机化的排除，保留成本及未知终局；pending 不自动重试，单臂不构造配对伤害标签。第二部分真实实验和自然/四格覆盖尚未完成；统计估计、策略训练、部署及 Inspect 均未实现。本轮只完成 S2-02 并停止。

## 2026-11-15 — S2-01 可恢复目标配对软件闭环

在 `ee97ac12d4e43d4f168e6ce2acb5f2196c1a56f5` 基础上使用分支 `codex/s2-paired-experiments`；两名实现 agent、一名独立审核 agent 协作。保留开始前所有未提交改动，未提交或推送代码。以下仅列本任务相对开始时工作区的新增/修改，不把已有采样、检索实现记作本轮新工作。

| 文件 | 本轮改动 |
| --- | --- |
| `src/cave_mem_v3/memory_generation/replay.py` | v6 来源规则要求所有修正成功及平均效用差超阈值；v5 保持原语义；目标入口检查新规则 K=2 |
| `src/cave_mem_v3/agents/runtime.py` | 修复选中检查点剩余步数重复扣减 |
| `src/cave_mem_v3/agents/providers.py` | target 专用瞬时提醒入口；source 入口仍拒绝提醒 |
| `src/cave_mem_v3/target_data/collection.py` | 冻结 exposure_rules 内容，供后续校验 |
| `src/cave_mem_v3/interventions/__init__.py` | 新目标干预模块 |
| `src/cave_mem_v3/interventions/inputs.py` | 读取 prepared，绑定来源/目标分区，重核采样、状态、预算和共享成本，保留排除分母 |
| `src/cave_mem_v3/interventions/reminder.py` | 从公开来源字段生成确定性、长度受限提醒 |
| `src/cave_mem_v3/interventions/execution.py` | Silent 缓存动作、Remind 单次重提、相同普通续跑预算、成本分项、异常及 pending |
| `src/cave_mem_v3/interventions/collection.py` | 实际检查点恢复/隔离审核、冻结随机顺序、独立实例、不可覆盖尝试/结果及事件链 |
| `src/cave_mem_v3/interventions/outcomes.py` | 只读验证原始事件，重算 D/H/rescue、部署费用及实验实际费用；未知结局/成本不补造 |
| `src/cave_mem_v3/cli.py` | `pair-target` 与 `summarize-target-pairs` 命令；pending 返回非零 |
| `examples/phase2_demo.py` | 无网络的来源验证 → prepared → paired 完整示例，拒绝覆盖 |
| `tests/test_evidence.py` | 新规则、旧证据兼容、通用 K 与目标 K=2 测试 |
| `tests/test_runtime.py` | 多种检查点位置与预算回归测试 |
| `tests/test_target_pair_inputs.py` | 分区、来源、预算、费用和重封后不一致输入测试 |
| `tests/test_target_execution.py` | 提醒、缓存动作、配对种子、恢复、计时及单臂异常测试 |
| `tests/test_target_pairs.py` | 闭环、成功/伤害/未知成本、pending、恢复运行、CLI、示例与审核回归测试 |
| `README.md` | 更新已实现范围与运行命令 |
| `docs/phase1.md` | 更新 v6 新规则与旧证据边界 |
| `docs/experiment_design.md` | 同步用户确认规则及当前配对实现范围 |
| `docs/task_cards.md` | S2-01 完成记录；原细分卡作为历史参考 |
| `CHANGELOG.md` | 本条逐文件交付及验证记录 |

验证：`python scripts/run_tests.py` 共 189 项软件测试通过，`git diff --check` 通过；独立审核无剩余阻断问题。运行 `python examples/phase2_demo.py --output runs/s2-01-fixture-20261115`，保存 1 个目标 rollout、1 对完成且有效的目标结果。配置为 fixture actor/encoder、4 步/4 次模型调用、来源 K=2、hazard=0.99、seed=42、模型调用成本权重 0.01。Silent 失败、Remind 成功，`D=0.97, H=0, rescue=1`；计入共享准备后模型调用分别为 1 和 4，整个实验实际共 4 次脚本 actor 调用（真实 API 请求为 0）。

限制：以上均为软件观测，不是模型或 benchmark 结论；真实 Terminal 运行仍需冻结配置和已审核的离线文件系统协议，恢复不保证远端 RNG 完全耦合。当前只执行 target_train 的配对试点；未实现单臂、训练、部署或 Inspect。fixture 无 finalizer 时，非终局预算耗尽保留未知结局。既有规则/预算不一致的 prepared 或旧 v5 库不被自动迁移，需要新的来源/准备产物。

## 2026-11-15 — K=2、20 步工程试跑协议

- `configs/phase1/server.toml`：历史轨迹和来源回放的 actor 预算均设为最多 20 步/20 次模型调用，K 保持 2；这一配置只用于新的工程 run，不能复用旧 6 步来源轨迹。
- `configs/phase1/demo.toml`：fixture 环境保留 4 步预算，配对回放改为 K=2，使完整软件流程更快完成。
- `src/cave_mem_v3/cli.py`、`tests/test_cli.py`：移除置信区间的候选家族参数注入，并按 K=2 检验端到端入库。
- `docs/phase1.md`、`docs/experiment_design.md`：明确工程 pilot 在全部 K 组有效后按平均效用差超过阈值赋 `v=1`；区别旧 r3 pilot 产物与正式研究证据，修正运行产物和预算说明。
- `src/cave_mem_v3/memory_generation/replay.py`、`src/cave_mem_v3/memory_generation/bank.py`、`tests/test_evidence.py`：另由回放实现 agent 修改，去掉置信区间入库门槛并保持来源恢复和异常校验。

历史 pilot 产物保持原样；新设置必须使用新的 run 名和冻结计划。

## 2026-11-15 — 第一阶段固定 K 回放

- `src/cave_mem_v3/memory_generation/replay.py`：移除筛选/确认双队列，改为每个冻结候选完整执行配置指定的 K 组配对回放；所有尝试保留，完成后按全部 K 组重算效用差置信区间。证据与计划 schema 升为 v4，只有下界超过 `acceptance_epsilon` 才标记 `v=1`。
- `src/cave_mem_v3/common/config.py`、`src/cave_mem_v3/cli.py`：要求显式设置 `verification.replay_pairs`，移除旧双阶段参数校验；整批 K 回放完成后自动将所有 `v=1` 项冻结到 verification 输出目录的 `bank.json`。
- `src/cave_mem_v3/memory_generation/bank.py`：记忆质量记录 K 对回放数；现有正式 `bank` 命令仍只冻结 `v=1` 候选。
- `configs/phase1/demo.toml`、`configs/phase1/server.toml`：改为单一 `replay_pairs` 参数；demo 使用 K=16，H20 配置中的 K=2 明确仅为工程 pilot，正式 K 尚待冻结。
- `tests/test_evidence.py`、`tests/test_cli.py`：覆盖固定 K 完整运行、缺对/重复对拒绝、错误保留、置信阈值入库及配置缺项。
- `docs/phase1.md`、`docs/experiment_design.md`：说明教师查看完整公开轨迹并提出检查点/修正假说；K 次回放过阈值后自动入正式库；明确 proposal 未指定 K，也未规定筛选/确认拆分。将旧 `pilot-provisional-bank.v1` 描述为历史工程产物。

当前正式 K、成本权重、接受阈值与任务范围审计仍需在正式来源实验前冻结；本次代码改动不把 H20 的 K=2 pilot 升格为正式研究证据。

## 2026-11-14 — H20 Terminal-Bench 工程 pilot

### 失败归因复核及新一轮采集

- `src/cave_mem_v3/agents/providers.py`：可显式请求 OpenAI-compatible `json_object` 输出，记录在冻结的 provider 配置与请求中；默认纯文本行为及单次计费不变。
- `tests/test_providers.py`：覆盖 actor JSON 请求、教师默认请求及非法配置。
- `configs/phase1/server.toml`：只为 Qwen actor 打开 `response_format="json_object"`；教师接口不变。该变动要求新的历史 run，不能与旧 checkpoint 混用。
- `src/cave_mem_v3/cli.py`：自动教师生成跳过未完成评分的无效动作轨迹，输出被排除的来源 ID 与原因；手工模板及正式评分不变。
- `tests/test_cli.py`：验证未解析动作不会发送给教师、会被显式记录为排除。
- `scripts/phase1_actor_service.py`：将一次性 H20 vLLM 启动/清理及采集或回放调度放进版本控制，显式不把教师密钥传给 actor 作业。
- `scripts/phase1_pilot_replay.py`：收录本轮 K=2 来源配对工程回放，输出只标作暂定库，不修改正式 `verify`/`bank`。
- `docs/phase1.md`：明确任务为事先有目的选择、旧轮程序/格式问题和官方测试失败的区分。

`runs/pilot-20261114-r2/` 的 3 条历史轨迹均未成功，教师 3 次请求完成并生成 3 条候选；各候选的 K=2 来源回放均未通过，暂定库为 0。`regex-log` 原始 actor 的 JSON 反斜杠转义错误，命令未执行；`fix-git` 官方测试 2 项失败；`sqlite-db-truncate` 中 actor 重复调用镜像内不存在的 `sqlite3`，教师候选又遇到截断数据库不可直接用 sqlite3 读取。Enroot 的 shadow 钩子警告混入 stderr，命令返回码和独立官方测试仍可用，但会污染 actor 观察，尚不能声称与原生 Harbor 会话等价。旧轮产物不改写；新配置从 `runs/pilot-20261114-r3/` 重新采集。

`r3` 在同一预冻结的 3 道 history 上重新采集：3 次运行、0 成功、3 次官方测试失败、0 基础设施错误、0 JSON 解析错误。actor 采集调用 18 次，输入/输出 token 为 19,524/862，执行 18 次 shell；教师 3 次调用均完成，生成 3 个候选，输入/输出 token 为 9,831/3,146，排除轨迹 0。两组来源配对回放每候选各 2 组，共 6 组；回放 actor 调用 31 次，输入/输出 token 为 45,257/1,396，执行 42 次 shell，0 异常、0 解析错误。`regex-log` 候选 `9ae9c24d...` 两组原动作失败/修正成功，效用差均 +1，进入 `pilot-provisional-bank.json`；`fix-git` `2f57b185...` 和 `sqlite-db-truncate` `789d03f6...` 两组均不通过并被排除。暂定库 1，**正式 `v=1` 库 0**。总模型调用 52 次、输入 74,612 token、输出 5,404 token；未配置单价，货币成本未知，不推算。H20 作业结束后 vLLM 进程退出。

### 修改文件

- `src/cave_mem_v3/environments/terminal.py`：描述从 Docker 独占改为容器后端通用；正式恢复判据未放宽。
- `src/cave_mem_v3/environments/enroot.py`：新增独立 SquashFS 镜像快照、actor 隔离、临时 tests 副本评分和 `/tmp` 动作边界同步；原 Docker 后端保留。
- `src/cave_mem_v3/cli.py`：终端任务可显式选择 Enroot，未选择时仍用 Docker；教师和 actor 仍使用不同配置。
- `src/cave_mem_v3/common/config.py`：`check` 显示本地模型目录是否存在，并声明不是权重版本证明。
- `configs/phase1/server.toml`：冻结本轮 H20 actor、教师网关别名、3 次历史采集与 6 步预算；本地 actor 无需 API key，教师密钥只从服务器环境读取。
- `tests/test_cli.py`：适配服务器 pilot 配置的只读检查。
- `tests/test_enroot.py`：检查 actor 的隔离参数和 `/tmp` 快照处理。
- `docs/phase1.md`：记录 Enroot 适用范围、7 道任务的角色分离、K=2 暂定库与正式 `v=1` 的差异。

### 未提交的服务器数据

- `data/terminal-bench-2/`：官方 v2 任务仓库（89 道任务），固定提交 `2fd12b88aafdd04a52c298e3940bcb189f9766d6`；每个子目录是一道任务，不是独立数据集。
- `data/images/`、`data/snapshots/`：Enroot 任务镜像和动作快照；`data/pilot_catalog.json` 预先冻结任务角色与镜像，脚本仅在 ignored `data/` 运行。
- `runs/pilot-20261114/`：首轮 3 条 actor 凭证缺失导致的基础设施错误，保留原始记录。
- `runs/pilot-20261114-r2/`：同一任务划分重跑的真实轨迹、自动候选、恢复诊断与来源回放；临时结果不冒充正式记忆库。

教师网关未提供可验证权重 revision；官方 verifier 联网取依赖，恢复审计仍标为未正式批准。K=2 暂定门槛不修改原版评分或 `v=1` 定义。

## 2026-11-13 — 第一部分接入独立教师配置

### 修改文件

- `.gitignore`：忽略服务器上的 `data/` benchmark 任务资产，避免将官方数据集和缓存误提交。
- `src/cave_mem_v3/cli.py`：自动候选生成改为读取独立 `[teacher]` provider；冻结 actor 仍只由 `[actor]` 驱动。
- `src/cave_mem_v3/common/config.py`：允许可选教师配置，并在存在时检查其模型、endpoint 和 revision。
- `configs/phase1/server.toml`：独立 teacher 采用用户给定的 `gpt-5.6-terra`、`https://api.aicodemirror.ai/v1` 和 `OPENAI_API_KEY` 环境变量；鉴权尚未通过，revision 仍未冻结，密钥不写入文件。
- `docs/phase1.md`：明确 actor/teacher 分离及密钥读取边界。
- `tests/test_cli.py`：更新自动候选 journal 测试，覆盖服务器模板的教师待配置状态。

候选仍是未验证假说，只有独立来源配对回放通过后才允许写入记忆库。

## 2026-11-13 — 按职责整理研究仓库

单 agent 完成。读取 GitHub 上 `huangwenzhe77/SkillRL` 的根目录、`agent_system/` 与 `skill_generation/`，采用其功能分组、示例和文档分工；CAVE 保留 `src` 包布局与 `cave3` 命令。此次改动发生在本地 CAVE-Mem-v3，未发布到 GitHub。

### 移动文件及导入更新

下表源码路径均相对 `src/cave_mem_v3/`。迁移后的模块同步更新导入；研究算法和数据 schema 不变。

| 原路径 | 新路径 | 职责 |
| --- | --- | --- |
| `contracts.py` | `common/contracts.py` | 共享数据、效用与持久化合同 |
| `config.py` | `common/config.py` | 共享配置加载与参数检查 |
| `providers.py` | `agents/providers.py` | 模型提供器与冻结 actor |
| `runtime.py` | `agents/runtime.py` | 计量后的单条轨迹执行 |
| `tasks.py` | `benchmarks/tasks.py` | 任务角色、依赖分组与资产校验 |
| `environments.py` | `environments/base.py` | 环境协议与开发 fixture |
| `terminal.py` | `environments/terminal.py` | Docker 任务执行与隔离评分 |
| `collection.py` | `memory_generation/collection.py` | 历史数据采集 |
| `candidates.py` | `memory_generation/candidates.py` | 失败修正提议 |
| `predicates.py` | `memory_generation/predicates.py` | 公开可执行条件 |
| `replay.py` | `memory_generation/replay.py` | 来源回放和独立确认 |
| `memory.py` | `memory_generation/bank.py` | 验证记忆库 |

以下配置和文档路径相对仓库根目录：

| 原路径 | 新路径 | 内容 |
| --- | --- | --- |
| `configs/phase1.demo.toml` | `configs/phase1/demo.toml` | 开发配置，参数保持不变 |
| `configs/phase1.server.toml` | `configs/phase1/server.toml` | 待填写服务器配置，参数保持不变 |
| `configs/tasks.demo.json` | `configs/benchmarks/demo.json` | fixture 任务清单 |
| `configs/tasks.server.example.json` | `configs/benchmarks/terminal.example.json` | 真实任务清单模板 |
| `EXPERIMENT_DESIGN.md` | `docs/experiment_design.md` | 四阶段实验规划 |
| 原 `README.md` 的详细内容 | `docs/phase1.md` | 第一部分操作指南，更新配置/源码路径和导航 |

### 新增文件

| 文件 | 内容 |
| --- | --- |
| `src/cave_mem_v3/agents/__init__.py` | actor 与执行子包定义 |
| `src/cave_mem_v3/benchmarks/__init__.py` | benchmark 子包定义 |
| `src/cave_mem_v3/common/__init__.py` | 公共合同子包定义 |
| `src/cave_mem_v3/environments/__init__.py` | 环境子包定义 |
| `src/cave_mem_v3/memory_generation/__init__.py` | 第一部分记忆构建子包定义 |
| `docs/architecture.md` | 目录方案比较、四阶段映射和后续扩展位置 |
| `examples/phase1_demo.py` | 复用现有 CLI 的完整 fixture 演示，保护已有输出 |
| `scripts/run_tests.py` | 不依赖当前工作目录的离线测试入口 |

### 同步修改

- `README.md`：重写为研究项目首页，展示目录树、四阶段状态、快速开始及文档导航。
- `src/cave_mem_v3/cli.py`：按新职责目录导入模块，保留现有命令和参数语义。
- `tests/test_candidates.py`、`tests/test_evidence.py`、`tests/test_predicates.py`、`tests/test_providers.py`、`tests/test_runtime.py`、`tests/test_tasks_collection.py`、`tests/test_terminal.py`：更新导入路径。
- `tests/test_cli.py`：更新导入路径与配置路径。
- `CHANGELOG.md`：记录本次移动、新增、修改及验证。下方历史条目保留当时路径，可用本节迁移表对照。

验证：Python 3.12 运行 `scripts/run_tests.py`，**77 项通过**；包发现正确识别主包及 5 个子包。`examples/phase1_demo.py --output runs/structure-demo` 完成全部第一部分流程，输出 4 条 fixture 轨迹、2 个修正候选及 2 条已验证记忆。它是开发演示，不构成真实 benchmark 结果。产物保留在被忽略的 `runs/structure-demo/`。

## 2026-11-13 — 第一部分实现

完成范围：proposal §3.1 历史失败记忆。三个实现 agent 分别负责任务采集、运行后端、来源证据；第四位独立 agent 审核代码并复核修复。主 agent 完成共享合同、命令集成、配置和文档。

### 新增源码

| 文件 | 新增内容 |
| --- | --- |
| `src/cave_mem_v3/__init__.py` | 包版本与第一部分范围说明 |
| `src/cave_mem_v3/__main__.py` | `python -m cave_mem_v3` 入口 |
| `src/cave_mem_v3/contracts.py` | PublicContext、StudySpec、SourceMemory、效用、哈希和持久化合同 |
| `src/cave_mem_v3/config.py` | TOML 加载、未确定 actor/采集/回放参数检查 |
| `src/cave_mem_v3/tasks.py` | 四类 benchmark 角色、验证集细分、连通组划分与资产校验 |
| `src/cave_mem_v3/collection.py` | 只采历史任务，固定次数或失败目标、每任务/总次数上限、保留所有结果和续跑校验 |
| `src/cave_mem_v3/providers.py` | 冻结 actor、Qwen 远程接口、响应解析、单次请求计量和实际响应身份核验 |
| `src/cave_mem_v3/environments.py` | 环境协议及可恢复的软件 fixture |
| `src/cave_mem_v3/terminal.py` | Terminal-Bench 文件系统协议的 Docker 后端、隔离评分及恢复诊断 |
| `src/cave_mem_v3/runtime.py` | 历史执行和来源修正续跑、动作前快照、费用与错误记录 |
| `src/cave_mem_v3/candidates.py` | 失败步骤绑定、模板/模型修正提议、真实失败签名与模型提议错误计量 |
| `src/cave_mem_v3/predicates.py` | 公开条件的可执行表达、输入校验和三值求值 |
| `src/cave_mem_v3/replay.py` | 预冻结筛选/确认计划、匹配两臂、来源响应身份核验、Eq.6 效用差与置信区间 |
| `src/cave_mem_v3/memory.py` | `m=(p-,a-,f,ã,P,N,E,q)`、仅验证记忆入库、不可覆盖主库与加载重核 |
| `src/cave_mem_v3/cli.py` | check、manifest、collect、candidates、audit、verify、bank 命令及候选生成持久化 journal |

### 新增配置与项目文件

| 文件 | 新增内容 |
| --- | --- |
| `configs/phase1.server.toml` | Qwen2.5-7B-Instruct 模型名；服务地址/revision/路径和未确定次数保留空缺；无下载行为 |
| `configs/tasks.server.example.json` | 历史/目标训练/验证校准/验证选择/测试任务清单占位模板 |
| `configs/phase1.demo.toml` | 开发 fixture 的明确小预算，仅用于软件演示 |
| `configs/tasks.demo.json` | 可直接执行的 fixture 任务角色示例 |
| `pyproject.toml` | 独立 Python 包和 cave3 命令，不依赖 v2 运行时 |
| `.gitignore` | 忽略环境、运行产物、模型权重和密钥 |
| `README.md` | 实验划分/停止规则解释、架构、安装、逐步命令及服务器使用边界 |
| `CHANGELOG.md` | 本逐文件同步记录 |

### 新增测试

| 文件 | 验证内容 |
| --- | --- |
| `tests/test_tasks_collection.py` | 分组泄漏、次数上限、成功/失败/错误保存、资产/身份变化和续跑 |
| `tests/test_runtime.py` | 公开/私有边界、恢复、预算、费用、初始化错误及身份漂移 |
| `tests/test_providers.py` | 无配置时无下载/联网、请求计量、格式错误、模型与指纹核验 |
| `tests/test_terminal.py` | 注入后端下的恢复、文件隔离、评分结果与基础设施错误分类 |
| `tests/test_candidates.py` | 失败来源绑定、条件/签名、生成提示隔离及已收费失败保存 |
| `tests/test_predicates.py` | 可执行条件、未知值、公开文本和禁止的私有字段 |
| `tests/test_evidence.py` | 独立确认、效用/区间重核、无效恢复、身份漂移、篡改和主库读写 |
| `tests/test_cli.py` | 全命令闭环、显式预算、空服务器配置、候选生成 journal 与无重复调用恢复 |

### 修改文件

`EXPERIMENT_DESIGN.md`：顶层数据划分明确为历史、目标训练、验证、测试；验证内部区分校准/选型。按用户纠正写入 `x_t` 的小写 `e_t`，撤回上轮对暴露含义的额外重定义，保留 proposal 原定义。原 PDF 和 CAVE-Mem-v2 未修改。

### 审核与验证结果

Python 3.12 下 `python -m unittest discover -s tests -v`：**77 项通过**。完整 CLI 链路在 fixture 上实现历史采集 → 候选 → 独立回放 → 冻结库 → 重新加载核验。服务器 `check` 如实列出尚待填写的 endpoint、revision、采集次数和筛选/确认对数。

独立审核发现并修复了每任务次数上限被忽略、未填 K 静默默认、初始化异常记录缺失、候选模型失败计量丢失、叙述条件不可执行及真实响应模型身份漂移。复核未发现未修复的 P1/P2 问题。

验证范围是软件测试及注入后端。没有下载模型，没有运行真实 Qwen 服务、Docker 容器或官方 benchmark，因而没有实证成功率/失败率或科研性能结论。
