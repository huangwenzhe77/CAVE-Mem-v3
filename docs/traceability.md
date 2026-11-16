# Proposal 与流程图到代码的追踪表

版本：2026-11-16；对应本地未发布分支 `codex/phase3-goal-workflow`，基线 `e3a6ef891f47a1cfd779eb710ed9bf0fd46bd398`。本表解释本轮代码改动如何对应 proposal 和用户提供的四张流程图。`Implemented` 表示软件路径存在并通过本轮离线软件检查；不表示研究假说已获支持或真实 benchmark 已运行。

## 需求编号

| ID | Proposal 对应 | 流程图节点/要求 | 代码验收含义 |
| --- | --- | --- | --- |
| R1 | §3.1、Eq. (6)，memory tuple Eq. (1) | 图1：运行 history、提出失败机制/修正/适用条件、原动作与修正动作做 K 次来源重放、比较收益并冻结已验证记忆 | 每条主库记忆必须关联原始轨迹、候选条件、K=2 独立配对证据；两组修正均成功、配对有效且均值收益严格超过 epsilon。教师意见不代替回放证据 |
| R2 | §2.1 Eq. (2)、§3.2、Algorithm 1 第 3–5 行 | 图2：目标任务到候选检查点；检索最多一个记忆；缓存未见记忆的 `a0`；在结局前评估 `v/ρ/e/d/b` 并冻结 `x`；再随机分 Silent/Remind | 特征只用干预前可见资料；`e_t` 是独立的动作暴露量，v/ρ/e 可未知。不得让未来臂结果影响 `a0/x` |
| R3 | §2.2 Eq. (3)–(5)、§3.3、Algorithm 1 第 6 行 | 图2：可恢复检查点执行同一检查点两臂，记录效用差 `D` 和配对回退 `H`；不可恢复检查点只执行随机分配的一臂并记录 propensity | 配对与单臂记录是不同统计对象。单臂没有已观察个体 `D/H/rescue`；恢复失败不降级成配对 |
| R4 | §3.3 要求训练/validation/test 按任务族、来源机制和目标模板隔离；§4.1 提到 validation calibration，§4.2 要求 full-task validation；proposal 正文并未规定 calibration/selection 必须是两个独立子集 | 图2至图4的数据依赖：图3提到校准相关评估和“验证集选择策略”，但没有单独画出 calibration 与 selection 两个数据分区；图4要求未见任务评估 | 代码额外把 validation 显式分成 calibration 与 selection 两种子角色，避免同一验证结果既校准模型又选策略。这是实现层的数据隔离约束，不是流程图逐字要求；最终 test 仍只经冻结计划授权 |
| R5 | §4.1 Eq. (7) | 图3左：配对数据训练收益 `τ(x)` 与风险 `h(x)`，以校准结果构造监督初始策略 `πSFT` | 收益、伤害、各成本头分别按有效标签掩码拟合；独立校准；用同容量 gate 作为监督基线 |
| R6 | §4.2 Eq. (8)–(9) | 图3右：估计 `μ0(x), μ1(x)`，用 DR 目标相对 Silent 评估/训练策略；配对直接目标与单臂 DR 分开 | 组级交叉拟合、propensity、权重/ESS/截断诊断；DR 是估计量，非伪造反事实标签；当前 DR bandit 的初始化仍需兼容配对模型/校准/gate |
| R7 | §4.3 Eq. (10)–(14) | 图3中下：优化提醒策略，加入回退风险上限 `δ` 和计算预算 `B` | 同容量 gate、KL 正则、组 bootstrap 经验上界和非负对偶更新；首版计算约束只对应 `selected_gate_actor_calls`，经验界不是安全定理 |
| R8 | Algorithm 1 第 8–14 行、§4.3 | 图4：训练/校准后冻结策略；新任务缓存 `a0/x`；证据不足/净收益不确定时退回 Silent；提醒后重验工具权限；未见任务评估 | 自包含冻结包、完整 validation rollout、实际策略后置保护规则、配对或保守选型锁；执行中不训练、不重采 `a0`、不自动重试 |
| R9 | §5.1–5.3、Table 1 | 图4：在未见任务比较成功率、回退、提醒、成本，并报告不同条件 | 最终 test 计划按依赖组分配策略；按完整任务分母报告边际/成本和组级诊断。`v/ρ/e` 检查点格子仅为描述性覆盖，不是随机子组因果估计 |
| R10 | §4.4 Eq. (15)、§5.2 | 图4可选 Inspect：独立三动作/两阶段随机数据和 VOI 策略 | 当前未实现。二动作 Silent/Remind 日志不能识别 Inspect 的信息价值 |

## 源码逐文件对应

### 第一、二部分及公共边界

| 改动文件 | 对应 ID | 本轮在代码中承担的工作 |
| --- | --- | --- |
| `src/cave_mem_v3/cli.py` | R2–R9 | 接通 dataset、训练、校准、DR、bandit、冻结、validation、两类选型和最终 test/report 命令；test 与 no_memory 路径在 provider 初始化前受授权门禁 |
| `src/cave_mem_v3/target_data/partitions.py` | R4 | 实现 train/validation/test 分区，并把 validation 进一步拆为 calibration/selection 子角色；后一个拆分是防止验证数据复用的工程约束，不是图3明确画出的节点 |
| `src/cave_mem_v3/target_data/collection.py` | R2、R4 | 在已有 prepared 记录中加入可审计的内部执行配置身份，供前瞻 selection 绑定及防旁路替换 |
| `src/cave_mem_v3/target_data/sampling.py` | R2、R4、R8、R9 | outcome-blind 检查点抽样；最终 test 必须由计划授权；增加 no_memory 专用禁用抽样器 |
| `src/cave_mem_v3/interventions/collection.py` | R3、R4 | 将验证子角色传入已有配对执行路径并保持数据身份 |
| `src/cave_mem_v3/interventions/execution.py` | R3、R8 | 增加独立冻结策略执行协议；缓存动作、双臂权限检查与旧配对合同保持兼容 |
| `src/cave_mem_v3/interventions/inputs.py` | R3、R4 | 为新执行/计划记录提供封存和只读检查合同 |
| `src/cave_mem_v3/interventions/outcomes.py` | R3、R4 | 重核新增执行模式的事件、标签、身份与成本，保持原始未知状态 |
| `src/cave_mem_v3/interventions/single.py` | R3、R4、R8、R9 | 单臂路径传递 test 授权；no_memory 模式不建 encoder、不检索/抽样记忆，普通 actor 使用完整调用预算 |
| `src/cave_mem_v3/interventions/single_outcomes.py` | R3、R4、R8、R9 | 单臂与 no_memory 新日志的只读重建；缺臂、pending、未知结果仍保留 |

### 第三部分：训练与策略优化

| 改动文件 | 对应 ID | 本轮在代码中承担的工作 |
| --- | --- | --- |
| `src/cave_mem_v3/policy/__init__.py` | R5–R9 | 建立 policy 子包入口；不额外执行实验逻辑 |
| `src/cave_mem_v3/policy/dataset.py` | R3–R6 | 从配对/单臂原始结果构建版本化只读训练视图；隔离特征、标签、分组、成本、概率及未知状态 |
| `src/cave_mem_v3/policy/features.py` | R2、R5–R7 | 把冻结的上下文/`a0`向量和结构字段编码成可重现输入；为 v/ρ/e/置信度保留未知语义 |
| `src/cave_mem_v3/policy/supervised.py` | R5 | 拟合配对收益、风险与分项增量成本预测头；保存训练来源、掩码、参数和模型身份 |
| `src/cave_mem_v3/policy/calibration.py` | R5、R7、R8 | 在独立 calibration 组拟合收益/成本预测范围和风险分箱上界；不足支持保持 unavailable |
| `src/cave_mem_v3/policy/gate.py` | R5、R7 | 以校准支持和预设规则生成监督目标，拟合同容量 logistic 初始 gate |
| `src/cave_mem_v3/policy/dr.py` | R3、R6 | 按组 cross-fit `μ0/μ1`，计算 DR policy value/relative-Silent diagnostic、权重和 ESS；不产生配对 H |
| `src/cave_mem_v3/policy/bandit.py` | R6、R7 | 分开执行配对直接目标或单臂 DR 目标；KL、经验风险/调用上界、对偶更新和同容量比较 |

### 第四部分：冻结、验证、选型和最终测试

| 改动文件 | 对应 ID | 本轮在代码中承担的工作 |
| --- | --- | --- |
| `src/cave_mem_v3/policy/frozen.py` | R8、R9 | 将规则/监督/bandit/no_memory 编为带来源审计的冻结包；执行时计算实际提醒概率并应用保守后置门禁 |
| `src/cave_mem_v3/policy/rollout.py` | R8、R9 | 在 validation/selection 完整任务上运行冻结策略，保留无记忆、无检查点、pending 和全任务成本 |
| `src/cave_mem_v3/policy/selection.py` | R7–R9 | 前瞻登记候选；完整 validation 结果加独立配对检查点风险/调用审核，生成不可覆盖锁 |
| `src/cave_mem_v3/policy/conservative_selection.py` | R3、R7–R9 | 给不可恢复环境单独的保守选型协议，以最坏 `h=1` 局部界审核，不降级成个体配对风险 |
| `src/cave_mem_v3/policy/final_rollout.py` | R4、R8、R9 | 仅运行冻结最终计划分配的 test 任务；不读历史训练结局，不自行补跑 |
| `src/cave_mem_v3/policy/final_evaluation.py` | R4、R9 | 封存最终选择锁和 test 组分配；重核身份/资产/执行并报告完整任务边际指标、成本和组级统计 |

## 测试和可运行示例对应

| 改动文件 | 对应 ID | 验证内容 |
| --- | --- | --- |
| `tests/test_policy_dataset.py` | R3、R4 | 原始记录重核、角色/分组隔离、未知与单臂不伪造配对标签 |
| `tests/test_validation_partitions.py` | R4 | train/calibration/selection/test 角色门禁和前置拒绝 |
| `tests/test_supervised_policy.py` | R5 | 已知数值关系、头部掩码、校准、gate、角色与身份失败行为 |
| `tests/test_single_dr.py` | R3、R6 | 手算 DR、交叉拟合隔离、propensity、未知结果、极端权重和截断 |
| `tests/test_bandit_policy.py` | R6、R7 | 目标/梯度、KL、对偶更新、预算和未知支持约束 |
| `tests/test_policy_rollout.py` | R8 | 冻结策略执行、完整分母、预算、权限、pending 与只读重入 |
| `tests/test_policy_selection.py` | R7–R9 | 前瞻计划、完整候选、配对审核、风险/调用界和锁重核 |
| `tests/test_conservative_selection.py` | R3、R7–R9 | 不可恢复最坏界、no_memory、未知/缺失和独立锁 |
| `tests/test_final_evaluation.py` | R4、R9 | test 授权、组分配、HT/ESS/bootstrap、身份漂移、旧报告兼容 |
| `tests/test_learning_integration.py` | R1–R9 | 从原始 fixture 到实际拟合模型、冻结选型、最终 test；no_memory 不编码/检索，完整成本保留 |
| `tests/test_single_execution.py` | R3、R8 | 已有单臂执行合同与新模式回归兼容 |
| `examples/four_phase_learning_demo.py` | R1–R9 | 无 tests 导入的原始 artifact 完整链路；实际拟合、选择学习策略并完成最终 fixture test |
| `examples/phase3_supervised_demo.py` | R3–R6 | 第一至三部分训练/校准接口演示；小样本不足状态如实保留 |
| `examples/phase4_validation_demo.py` | R8–R9 | 冻结基线的完整 validation 演示，不代替赢家或真实 test |

## 文档、元数据与项目入口

| 改动文件 | 对应 ID | 文档职责 |
| --- | --- | --- |
| `README.md` | R1–R9 | 首页总览、命令入口、阶段状态、软件 fixture 与未发布提醒 |
| `AGENTS.md` | R1–R10 | 后续开发和研究边界，禁止将 fixture 当结果 |
| `docs/architecture.md` | R2–R9 | 阶段与模块职责索引 |
| `docs/phase1.md` | R1–R9 | 同步各阶段实现状态，不更改来源科学规则 |
| `docs/experiment_design.md` | R1–R10 | 保留 proposal 研究问题与限制，更新软件实现状态 |
| `docs/policy.md` | R3–R7 | 训练/估计范围、公式语义、预算和纯单臂限制 |
| `docs/evaluation.md` | R7–R10 | 冻结验证、两种选择协议、test 报告及 no_memory/Silent 区别 |
| `docs/software_goal.md` | R1–R10 | 总体软件目标、proposal/流程图阶段映射和完成边界 |
| `docs/software_handoff.md` | R1–R10 | 服务器安装/准备、数据/身份前置、命令、真实采集下一步和研究限制 |
| `docs/server_handoff.md` | R1–R10 | 保留旧发布快照并链接本地新交接，避免误认远端含有本轮代码 |
| `docs/task_cards.md` | R1–R9 | 逐卡目标/规则/验收和逐卡实施记录 |
| `CHANGELOG.md` | R1–R9 | 本轮逐文件变化、测试设置、fixture 输出和未完成研究事项 |
| `pyproject.toml` | R5–R9 | 新增 NumPy policy extra 并更新包功能描述 |

## 四张图逐阶段结论

| 流程图 | 已对应的软件能力 | 未由本轮软件验收证明的部分 |
| --- | --- | --- |
| 图1：构建历史失败记忆 | 既有 history/candidate/replay/bank 模块；I-01 示例重新走一遍 fixture K=2 | 真实历史任务、教师输出和真实环境恢复/评分没有执行 |
| 图2：收集目标提醒数据 | 既有 a0/x 冻结、可恢复配对、不可恢复在线随机单臂；本轮补 validation/test 角色接口 | 真实任务 checkpoint、actor 质量、自然分布和四格/噪声条件没有采集；单臂从零训练未打通 |
| 图3：训练和选择策略 | 本轮 P3 数据视图、收益/风险/成本、校准、监督 gate、DR、约束 bandit 与 validation 选型 | 正式 δ/B/奖励权重/校准阈值未预登记；预算首版只约束局部 actor call；组界不是总体保证 |
| 图4：部署与最终评估 | 本轮 P4 冻结包、保守回退、工具权限复验、独立 test 计划与完整任务报告 | 真实未见 benchmark、第二档 actor、外部原系统强基线和统计结论未运行；Inspect 不实现 |

**图3与验证子集的说明：** 图上写的是收益/风险预测器、监督初始策略、DR/策略优化、加入 `δ/B` 并用验证集选策略；没有画出独立的 calibration 子集。proposal 正文 §4.1 明确提到用 validation 校准预测器，§4.2 又要求用 full task rollouts 做 validation。软件把这两种验证用途固化为两个互斥子角色，是为避免数据复用而作的实现选择；不能把它说成流程图明示的独立数据集要求。

## 贯穿全链路的可核验证据

- 根 agent 全量软件套件：353 项通过（2026-11-16）；实现新增 7 项 I-01 测试；两位独立 reviewer 对研究数据边界和工程分别通过。
- 最终 fixture：7 个收益/风险/成本头与监督 gate 实际 fitted，配对和单臂 bandit 均产生候选；新 validation 选中 paired-bandit；最终 12/12 个 test fixture 有结局。学习策略对 7 个分配任务作出决策、提醒 3 次。原始来源、配对和单臂 JSON 未改变。
- Python fixture、数字、超参数和输出路径仅供软件验收，不能支持 H1–H4 或任何真实收益声明；当前代码在本地分支，尚未提交/推送。
