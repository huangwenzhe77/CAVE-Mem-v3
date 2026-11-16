# 项目架构与目录选择

## 为什么采用这种目录

参考仓库为 [huangwenzhe77/SkillRL](https://github.com/huangwenzhe77/SkillRL)。其 `agent_system/`、`skill_generation/` 使功能职责直接可见，`examples/`、`scripts/`、`docs/` 分别承担复现实验、辅助操作与文档导航。CAVE-Mem-v3 采用这些职责分工。

| 组织方式 | 优点 | 对本项目的影响 |
| --- | --- | --- |
| 原来的 `src/cave_mem_v3/*.py` 平铺 | 规模小时路径短，安装简单 | 第一部分已有十多个模块，后续三部分会让导航困难 |
| SkillRL 式多个顶层功能包 | GitHub 首页直接展示功能，适合大型研究仓库 | CAVE 当前规模较小，公共数据合同和四阶段代码仍适合统一命名空间 |
| **`src` 包布局 + 内部职责目录** | 标准安装、清晰命名空间、逐阶段扩展，并有独立文档与示例 | **当前采用** |

这次保留 `cave_mem_v3` 包名及 `cave3` 命令，把实现文件按职责分组。新增子包均有实际实现；后续阶段的目录在对应代码开始实现时建立。

## 第一部分到文件的对应

所有源码路径以下均相对于 `src/cave_mem_v3/`。

| 职责 | 文件 | 输入 → 输出 |
| --- | --- | --- |
| 任务划分 | `benchmarks/tasks.py` | benchmark catalog → 冻结 manifest |
| 历史采集 | `memory_generation/collection.py` | history 任务、采集预算 → 全部历史轨迹 |
| 模型接口 | `agents/providers.py` | 公开上下文 → 动作与模型调用记录 |
| 轨迹执行 | `agents/runtime.py` | actor、环境、预算 → 单条轨迹/续跑 |
| 环境合同与开发环境 | `environments/base.py` | 动作 → 公开观察与检查点 |
| 真实任务后端 | `environments/terminal.py` | 固定任务资产 → 隔离执行、恢复与评分 |
| 修正提议 | `memory_generation/candidates.py` | 历史失败与成功参照 → 可证伪候选 |
| 适用条件 | `memory_generation/predicates.py` | 公开观察与条件 → `0/1/?` |
| 来源验证 | `memory_generation/replay.py` | 冻结候选与检查点 → 独立回放证据 |
| 记忆库 | `memory_generation/bank.py` | 已验证来源证据 → 不可覆盖的主库 |
| 公共数据合同 | `common/contracts.py` | 数据定义、成本与效用、序列化、哈希 |
| 公共配置 | `common/config.py` | TOML → 校验配置和待填写项 |
| 统一操作入口 | `cli.py`、`__main__.py` | 命令参数 → 上述各步 |

核心数据流为 `manifest → history traces → candidates → replay evidence → bank`。`examples/phase1_demo.py` 调用已有命令串起开发演示；`scripts/run_tests.py` 调用既有测试，不另行实现算法。

## 配置、文档和产物

- `configs/phase1/demo.toml`：可执行的软件演示预算。
- `configs/phase1/server.toml`：待填写的 Qwen 服务与实验预算。
- `configs/benchmarks/demo.json`：四角色的开发任务清单。
- `configs/benchmarks/terminal.example.json`：真实任务清单占位模板。
- [phase1.md](phase1.md)：第一部分完整使用指南，所有命令从仓库根目录运行。
- [experiment_design.md](experiment_design.md)：四阶段实验设计与研究边界。
- [../CHANGELOG.md](../CHANGELOG.md)：逐文件迁移、后续修改和验证记录。
- `runs/`：实际运行时创建，保存冻结计划、轨迹、证据和记忆库，不提交大体积运行数据。

## 后续阶段的扩展位置

第二部分的采样、检索与冻结特征已在 `target_data/`，可恢复配对和在线随机单臂采集已在 `interventions/`。第三部分 `policy/` 已实现数据入口、冻结特征向量化、配对模型、独立校准、监督 gate、单臂 DR 与约束 bandit；第四部分已实现冻结包、完整任务验证、配对/保守选型与独立最终 test，整体离线验收见任务卡。

公共模型与环境代码继续放在 `agents/` 和 `environments/`；第一部分的 `memory_generation/` 只负责来源记忆。目标提醒的观测结果已由第二部分采集代码保存；训练标签仅来自审核过的第二部分记录，最终测试结果只属于后续冻结评估；二者不能倒流为来源采集或原动作的输入。

## 第三部分模块

| 职责 | 文件 | 输入 → 输出 |
| --- | --- | --- |
| 审核数据视图 | `policy/dataset.py` | prepared/paired/single → 独立训练/校准/选型数据集 |
| 冻结特征 | `policy/features.py` | 已记录 x → 有明确版本/缺失语义的数值列 |
| 配对监督 | `policy/supervised.py` | target_train → τ/h/各增量成本预测器 |
| 校准 | `policy/calibration.py` | 独立 calibration → 组预测范围与风险分箱上界 |
| 初始策略 | `policy/gate.py` | 校准支持和训练 x → 同容量监督 logistic gate |
| 单臂估计 | `policy/dr.py` | 随机单臂 → 组交叉拟合与 DR/权重诊断 |
| 约束优化 | `policy/bandit.py` | 初始 gate 与配对/DR → KL/风险/调用预算约束候选 |

模型、校准和候选均是可核查 JSON。训练入口重读原始产物；预测入口只用冻结参数和 pre-treatment x。数值库通过可选 `policy` extra 安装，基础采集模块仍不依赖 NumPy。具体命令与统计范围见 [policy.md](policy.md)。

## 第四部分模块

| 职责 | 文件 | 输入 → 输出 |
| --- | --- | --- |
| 编包与决策 | `policy/frozen.py` | 已审核候选或规则 → 自包含冻结包与实际提醒概率 |
| 完整验证任务 | `policy/rollout.py` | selection 任务与冻结包 → 全任务轨迹、结局、费用 |
| 配对选型 | `policy/selection.py` | 前瞻候选计划、完整 rollout 与独立配对审核 → 选择锁 |
| 不可恢复选型 | `policy/conservative_selection.py` | 完整 live rollout、最坏风险/调用界 → 保守选择锁 |
| 最终计划与报告 | `policy/final_evaluation.py` | 选择锁、test 组分配 → 自包含最终计划与边际统计 |
| 最终执行 | `policy/final_rollout.py` | 已授权的 test 任务 → 单次完整执行与原始事件 |

训练/校准/选型加载时重核原始证据；最终冻结计划保存自包含快照，执行和报告不再读取历史训练结果。真实 test 访问由计划、任务和策略身份共同授权。全任务成功/效用和成本，与选中检查点的风险/调用预算是不同统计范围，不能混称。操作与限制见 [evaluation.md](evaluation.md)。
