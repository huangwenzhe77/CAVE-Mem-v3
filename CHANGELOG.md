# 变更记录

## 2026-11-16 — I-01 四部分核心软件整体验收

一名实现 agent 独占源码/测试修改，两名独立只读 agent 分别审核研究边界和工程正确性；主 agent 运行最终集成检查并维护文档。P3/P4/I-01 共 10 张代码卡按前置依赖逐卡完成，没有并行实施下一张卡。

| 文件 | 本卡变更 |
| --- | --- |
| `examples/four_phase_learning_demo.py` | 新增不导入 tests 的实际拟合完整示例；原始来源/配对/单臂→模型/校准/gate/两条bandit→新完整验证选择→最终test；原始证据hash守恒 |
| `tests/test_learning_integration.py` | 7项实际拟合/冻结执行/no_memory无检索与完整预算/pending/旧报告/未知覆盖测试；模块化helper避免重复收集测试 |
| `src/cave_mem_v3/policy/frozen.py` | no_memory独立协议身份；已有组件和学习包保持原协议 |
| `src/cave_mem_v3/target_data/sampling.py` | OrdinaryActorSampler显式关闭检查点抽样，保留角色和最终test授权 |
| `src/cave_mem_v3/interventions/single.py` | 仅冻结no_memory路径使用普通actor全预算，无encoder/检索/特征/gate/提醒；原公开单臂采集不变 |
| `src/cave_mem_v3/interventions/single_outcomes.py` | 普通actor禁用采样记录的只读重核，保持旧原始日志路径 |
| `src/cave_mem_v3/policy/rollout.py` | 无记忆完整验证无需encoder，保留所有任务、真实成本、终局与pending |
| `src/cave_mem_v3/policy/final_rollout.py` | 无记忆最终分配任务采用同一完整预算路径，真实test角色与授权不变 |
| `src/cave_mem_v3/policy/selection.py` | 配对选型识别no_memory结构性零干预风险/局部gate开销，完整任务费用不归零 |
| `src/cave_mem_v3/policy/conservative_selection.py` | 保守选型同样支持no_memory，无特征任务仍保留全任务和支持分母 |
| `src/cave_mem_v3/policy/final_evaluation.py` | 新计划登记v/ρ/e描述性覆盖，新report.v2保留未知决策/结局/费用；旧计划继续v1 |
| `src/cave_mem_v3/cli.py` | 独立no_memory验证跳过encoder构造；help总说明覆盖四部分 |
| `pyproject.toml` | 包描述更新为四部分软件范围 |
| `AGENTS.md` | 记录软件目标已验收、真实pilot另行授权，持续保留一写两审和逐卡规则 |
| `README.md` | 四部分状态、安装/完整演示入口、当前分支未发布及实际配置检查范围 |
| `docs/architecture.md` | 补齐第三四部分模块、两种选择协议和最终计划/执行/报告映射 |
| `docs/phase1.md` | 同步其他阶段的软件实现状态，不改来源研究规则 |
| `docs/experiment_design.md` | 更新角色/单臂统计实现状态，保持研究假说与真实实验边界 |
| `docs/policy.md` | 更新已实现后续模块，明确纯单臂从零尚不支持及预算/校准范围 |
| `docs/evaluation.md` | no_memory与Silent区别、结构性零的局部范围、v2描述覆盖与v1兼容 |
| `docs/software_goal.md` | 四部分与proposal对应、最终验收和剩余研究范围 |
| `docs/software_handoff.md` | 新交接：版本未发布、安装、完整演示、配置/数据迁移条件、限制、下一张R-01卡 |
| `docs/server_handoff.md` | 旧s2已发布版本保留历史内容，顶部链接新交接并明确旧状态不适用于新实现 |
| `docs/task_cards.md` | I-01及总体软件目标完成；真实R-01未启动 |
| `CHANGELOG.md` | 本表与实际验收、设置、产物和未完成事项 |

验收：实现新增 **7项**、工程独立 **66项**及完整演示、研究独立 **7项**及额外配对no_memory探针全部通过。根 agent `.venv/Scripts/python.exe scripts/run_tests.py` **353项通过（245.475秒）**；源码/测试/示例/配置的88份hash在最终测试启动后未改变。标准 `pip install -e '.[policy]'`、CLI帮助、文档链接与diff检查通过。使用Python3.14.8、NumPy2.5.4；最初无构建隔离的安装因虚拟环境缺setuptools失败，标准构建隔离安装成功，不涉及代码修复。当前server.toml只读检查是ready:true/model_downloaded:false，仅说明必填字段齐备，不是模型/资产/协议就绪证明。

根 agent 独立运行 `examples/four_phase_learning_demo.py --output runs/i-01-fixture-final-20261116`：1个history来源、24个train、12个calibration、8个selection、12个test任务；固定脚本actor/encoder、steps/calls4、seed42、hazard .999999、来源K2；校准alpha .2/min_groups2/bins2，gate风险阈值.9，paired/single bandit风险1/调用5/40轮/30bootstrap，单臂分配.5、DR两折，最终组分配.5/.5、seed42/bootstrap43。以上为明确的任意软件输入，不是研究参数建议。

实际结果：7个预测头与初始gate均fitted，paired/single bandit均candidate，DR24行/24组完整；新完整验证选中paired-bandit；最终12/12任务结局已知且evaluation_valid=true。学习策略实际分配7任务、决策7次、提醒3次，有效提醒概率范围[0,.9914369772466252]；no_memory分配5任务。源历史/回放、配对和单臂原始JSON hash不变，选择锁和最终报告重核一致。开发期dev01失败（示例响应字段名错误）与dev02旧协议演示均原样保留，不能当作最终兼容验收产物。全部为离线fixture，0真实模型/容器/benchmark请求。

范围：no_memory确实不检索/编码/构造记忆特征/gate/预留提醒槽，普通actor成本完整计入；结构性零只指干预风险和局部gate开销。v/ρ/e是选中检查点描述，未完成决策仍未知。纯单臂DR统计可独立运行，但单臂bandit/freeze仍需兼容配对model/calibration/warm_gate；纯单臂从零初始化/校准未实现。紧凑特征、局部actor调用预算、组级诊断界、共用来源库、最终单臂无个体H/rescue等限制保持明确。真实pilot/自然与四格/噪声库/第二actor/外部原系统复现/Inspect仍未完成。代码未提交、未推送，服务器需取得完成版本后另行准备真实运行。

## 2026-11-16 — P4-04 最终冻结 test 软件路径

| 文件 | 本卡变更 |
| --- | --- |
| `src/cave_mem_v3/policy/final_evaluation.py` | 两类选择锁编为自包含最终计划、组分配/身份/资产/顺序/目录冻结；完整task指标、HT、ESS、组bootstrap与分层，身份未知停用推断 |
| `src/cave_mem_v3/policy/final_rollout.py` | 独立真实test角色/原始协议，指定任务的活会话执行与重核，运行不读历史结局 |
| `src/cave_mem_v3/policy/rollout.py` | 私有会话重核支持明确test schema，默认validation不变 |
| `src/cave_mem_v3/target_data/sampling.py` | test采样需已验证最终计划与任务/策略授权，旧默认仍拒绝test |
| `src/cave_mem_v3/interventions/single.py`、`single_outcomes.py` | 活前缀及只读重核传递相同最终授权 |
| `src/cave_mem_v3/cli.py` | plan-final-evaluation、collect-final-evaluation、report-final-evaluation；无效/未完成非零退出，完整重读零provider |
| `tests/test_final_evaluation.py` | 14 项双锁、原始来源搬走仍可执行、组分配/HT、角色/预算/未知/身份/pending/CLI测试 |
| `docs/evaluation.md`、`docs/task_cards.md`、`CHANGELOG.md` | 最终协议、统计与边界、验收记录 |

两项审核通过：工程44项及独立CLI/HT复核；研究14项、追加2项和身份漂移探针。实现78项相关检查通过，根 agent 最终全量 **346 项通过（191.183 秒）**。根 agent 在 `runs/p4-04-fixture-20261116/` 建立最终计划并执行13个fixture任务（12个独立组，一组2任务），两策略概率 .5/.5，allocation42/bootstrap43，30 bootstrap、min2组，完整观察13任务并重核报告一致。角色确为test，所有数据仍是软件fixture，0外部请求。

未知分配结果/费用保留，正式总体指标不补零；不完整或未知/漂移模型身份停用总体/分层推断，保留描述。HT可能越出[0,1]不剪裁；组bootstrap不是精确随机化或多重比较保证。实际按策略批次执行顺序已冻结，外部时间影响不被自动排除。验证的风险证据不冒称test H。学习策略完整原始链路与no_memory基线由最后I-01补齐；未运行真实测试/提交/推送。

## 2026-11-16 — P4-03 不可恢复保守选型

| 文件 | 本卡变更 |
| --- | --- |
| `src/cave_mem_v3/policy/conservative_selection.py` | 独立前瞻计划/完整live候选/最坏风险与调用界/不可覆盖选择锁，不调用目标恢复 |
| `src/cave_mem_v3/policy/selection.py` | 配对协议同步修复逐响应实际身份未知门禁；未知候选或审核不能选出赢家 |
| `src/cave_mem_v3/cli.py` | plan-conservative-selection、collect-conservative-selection、select-conservative-policy |
| `tests/test_conservative_selection.py` | 10 项非恢复、公式、分母、未知/不可行/旧结果替换/身份与 CLI 测试 |
| `tests/test_policy_selection.py` | 追加两项部分未知live身份与配对审核未知身份回归 |
| `docs/evaluation.md`、`docs/task_cards.md`、`CHANGELOG.md` | 明确保守协议与实际身份门禁、验收记录 |

研究8项及增量4项/零调用预算探针、工程45项相关检查通过；根 agent 最终全量 **332 项通过（113.592 秒）**。`runs/p4-03-fixture-20261116/` 中3个独立任务、两个冻结候选完成非恢复完整验证，宽松fixture预算下选择remind；锁当前代码重核一致。旧配对示例锁同样重核通过。无目标恢复接口调用，0外部模型请求。

风险使用实际π的h=1最坏界；调用使用原始草稿1+π×剩余调用界，非观测H/反事实费用，也非完整任务费用上界。每个实际响应 model/fingerprint 必须完整一致，未知阻止全计划选型，不借其他候选已知身份补值；真实服务缺指纹仍需解决，不能补造。旧完整身份锁兼容，未知旧锁保留但不能冒称新门禁通过。test尚未执行。

## 2026-11-16 — P4-02 独立完整验证选型

| 文件 | 本卡变更 |
| --- | --- |
| `src/cave_mem_v3/policy/selection.py` | 前瞻计划、所有候选完整验证、内部计划绑定、独立配对风险/调用审核、固定排名、不可覆盖选择锁及重核 |
| `src/cave_mem_v3/policy/frozen.py` | 冻结包保存训练/校准原始目录保护范围 |
| `src/cave_mem_v3/target_data/collection.py` | 可选 config_identity 在 prepared run 哈希前写入，旧默认保持原形态 |
| `src/cave_mem_v3/cli.py` | plan-policy-selection、collect-policy-selection、select-policy |
| `tests/test_policy_selection.py` | 16 项完整选择、无可行/缺失阻断、路径与身份、风险零事件、旧审核替换、选择锁测试 |
| `docs/evaluation.md`、`docs/task_cards.md`、`CHANGELOG.md` | 协议、统计解释、验收与下一卡 |

两项独立审核通过：工程最终47项相关测试，研究15项及最终新增差异2项。根 agent 最终全量 **320 项通过（88.278 秒）**。根 agent 使用非论文fixture原始轨迹形态，在 `runs/p4-02-fixture-20261116/` 冻结2候选、每候选3独立任务、risk_limit=1、调用预算5、30 bootstrap；全部完整观察，脚本 remind 胜出，选择锁重核一致。仅验证软件，没有以测试阈值作研究选择，0外部模型请求。

修复了上下文样本零提醒被误称全局零风险、未完成候选被跳过，以及旧配对产物可通过旁边绑定文件替换的问题。现在原始 prepared 哈希绑定计划；全登记候选缺证据则无赢家，只有真正恒Silent规则可结构性零风险。bootstrap/有限组修正是有假设的经验诊断；配对审核是独立检查点证据，非live个体反事实。当前协议需要恢复，因此下一卡提供不可恢复的独立保守方案；test尚未开放。

## 2026-11-16 — P4-01 冻结包与完整验证任务

| 文件 | 本卡变更 |
| --- | --- |
| `src/cave_mem_v3/policy/frozen.py` | 自包含冻结规则/学习包，编包审计，实际策略/后置规则与原始有效概率，身份绑定 |
| `src/cave_mem_v3/policy/rollout.py` | selection 完整活会话、所有计划分母、局部判断耗时、journal/pending、原始日志重核及零调用重入 |
| `src/cave_mem_v3/interventions/execution.py` | 独立 opt-in frozen policy 执行模式；零剩余调用可执行缓存 a0，旧协议不变 |
| `src/cave_mem_v3/interventions/single_outcomes.py` | 新执行模式对应的只读验证；旧单臂默认校验不变 |
| `src/cave_mem_v3/cli.py` | freeze-policy、collect-policy-validation、summarize-policy-validation，完整重入在模型加载前检查 |
| `tests/test_policy_rollout.py` | 17 项完整分母/预算/记忆/未知/权限/瞬时提醒/冻结身份/重入/篡改测试 |
| `examples/phase4_validation_demo.py` | 冻结 Silent/Always-remind 后运行完整验证任务；无赢家或 test |
| `docs/evaluation.md`、`docs/architecture.md` | 冻结评估行为与现有第三部分模块说明 |
| `docs/task_cards.md`、`CHANGELOG.md` | 验收与下一卡 |

研究15项检查及最终工程48项相关检查通过；根 agent 全量 **304 项通过（57.893 秒）**。`examples/phase4_validation_demo.py --output runs/p4-01-fixture-20261116` 完成两个策略，每个1任务，均有观察终局；脚本 Silent失败、Always-remind成功，不选赢家。0外部请求，全部 fixture，未读取 test。

gate/后置规则/提醒构造耗时独立记录且计入一次；无记忆/R0缓存动作仍执行，未选中任务正常完成。原始配置、能力、协议、事件和费用重核，pending不重试，CLI重入不加载encoder/provider。学习证据不足的包可以明确强制 Silent，不能声称训练成功。相同后置规则可能改变训练策略成本，训练经验可行性不继承；单活会话不产生 H/rescue，下一卡另接独立匹配审核。

## 2026-11-16 — P3-05 风险与调用预算约束 bandit

| 文件 | 本卡变更 |
| --- | --- |
| `src/cave_mem_v3/policy/bandit.py` | 同容量二元策略、配对/DR 分别优化、KL、固定组 bootstrap 上界/梯度、非负对偶更新、无风险对照、实际草稿调用审计、掩码分母与候选验证 |
| `src/cave_mem_v3/cli.py` | train-bandit 显式配置与输入，输出训练状态/候选 |
| `tests/test_bandit_policy.py` | 13 项梯度、KL、约束抑制、掩码、成本、未知、篡改与原始产物 CLI 测试 |
| `docs/policy.md`、`docs/task_cards.md`、`CHANGELOG.md` | 约束单位/语义、验收与下一卡 |

一名实现、两名审核；各自 13 项针对性及独立有限差分/对偶/成本检查通过，最大梯度误差小于 1.8e-10。最终全量 **287 项通过（45.801 秒）**，diff 检查通过。主 agent 运行真实产物形态的 CLI 到 `runs/p3-05-fixture-20261116/candidate.json` 并重新加载一致：配对、风险限 .1、selected_gate_actor_calls 预算2、10轮/10 bootstrap；初始 gate 支持不足返回 unavailable/warm_gate_unavailable，未假造候选。有效拟合与约束抑制另由已知数值关系测试验证，全部为软件检查，0 外部请求。

首版预算只约束选中草稿的一次 actor 调用加提醒增量续跑调用，不包含完整前缀、Silent 基础续跑、token、耗时、费用或本地计算。配对 Δcalls 允许负值；单臂采用剩余调用保守上界及 h=1 最坏情况界，不伪造反事实。资格不足行保留 c0 与分母，未知记录阻止生成候选。只有最后迭代的经验可行性，未按验证集选赢家，没有部署授权或统计安全保证。

## 2026-11-16 — P3-04 单臂交叉拟合与 DR

| 文件 | 本卡变更 |
| --- | --- |
| `src/cave_mem_v3/policy/dr.py` | 按实际依赖组固定折叠，折内归一化与 μ0/μ1；式 8 价值、相对 Silent 改进、伪效用、原始/截断权重、ESS、分母及只读重建 |
| `src/cave_mem_v3/cli.py` | estimate-single-dr 入口，显式读取冻结诊断参数 |
| `tests/test_single_dr.py` | 12 项手算/数值、分折隔离、未知、缺支持、概率、极端权重与 API/CLI 测试 |
| `docs/policy.md`、`docs/task_cards.md`、`CHANGELOG.md` | 单臂统计解释与验收记录 |

一名实现、两名独立审核；两位审核各自 12 项测试及额外手算/重复组/极端权重检查通过。最终全量 `.venv/Scripts/python.exe scripts/run_tests.py` **274 项通过（40.964 秒）**。主 agent 从旧单臂原始数据通过 CLI 生成并重读 `runs/p3-04-fixture-20261116/dr-report.json`：p=.5、2 folds、每臂至少2组、seed42、L2=.01、截断10/100；一个计划样本保留，因仅一个独立组而 unavailable，价值 None，未补造对照臂或 H。该演示全为软件 fixture，0 外部模型请求。报告元数据后续补充缺失机制说明，较早演示按其原内容保留；最终代码另生成 `dr-report-final.json` 并重核一致，仍为不足支持状态。

限制：等组可用检查点诊断，不是自然任务总体价值；完整案例选择可能破坏随机化，需要缺失机制假设。截断偏移不是识别出的偏差界；配置先于加载并不证明事先未看结果；自适应训练策略仍需独立完整 rollout。单臂不产生已观测个体伤害，无真实实验。

## 2026-11-16 — P3-03 配对建模、独立校准和初始 gate

| 文件 | 本卡变更 |
| --- | --- |
| `src/cave_mem_v3/policy/features.py` | 冻结向量与结构化列契约，三值/置信度分类、缺失标记、维度/数值校验 |
| `src/cave_mem_v3/policy/supervised.py` | 按头掩码与组权重拟合 ridge/logistic，训练归一化、数值收敛、身份和 JSON 模型 |
| `src/cave_mem_v3/policy/calibration.py` | 独立校准，按实际组轴查泄漏，预测范围与风险分箱上界，支持不足及统计重核 |
| `src/cave_mem_v3/policy/gate.py` | 基于校准收益/风险为训练 x 生成目标，拟合同容量初始 gate，不足支持 Silent |
| `src/cave_mem_v3/cli.py` | train-policy、calibrate-policy、train-initial-gate、predict-policy |
| `tests/test_supervised_policy.py` | 13 项数值、掩码、角色、身份、支持、篡改、依赖和 API/CLI 集成测试 |
| `examples/phase3_supervised_demo.py` | 第一二部分原始 fixture 产物到第三部分模型/校准/gate 的离线链路 |
| `pyproject.toml` | 新增可选 policy NumPy 依赖 |
| `docs/policy.md` | 使用命令、模型/校准含义与统计限制 |
| `docs/server_handoff.md` | 标明旧已发布分支交接的历史范围，避免误认本地新代码已上传 |
| `README.md`、`docs/task_cards.md`、`CHANGELOG.md` | 状态、命令链接、验收和下一卡记录 |

一名实现、两名审核。研究/工程各自 13 项针对性测试与独立数值/反例检查通过；主 agent 使用 `.venv/Scripts/python.exe scripts/run_tests.py` **262 项通过（40.379 秒）**，diff 检查通过。Python 3.14.8 / NumPy 2.5.4，NumPy 已安装于项目忽略的 `.venv`，实现验证期间系统 Python 也安装了该版本。`python -S -m cave_mem_v3 --help` 在无 site-packages 下可用，调用建模函数明确提示安装 policy extra。

根 agent 运行 `examples/phase3_supervised_demo.py --output runs/p3-03-fixture-20261116`，所有产物保留，小样本各头与 gate 为 unavailable，最终 Silent/warm_gate_unavailable；外部模型请求 0。真实拟合能力另外由已知关系测试验证，不能把此示例当作有效研究模型。未运行真实采集或选择研究阈值。

限制：线性/logistic 轻量特征基线；收益/成本是实现值预测范围，不是条件均值置信区间，风险分箱组级上界不是逐点安全保证；等组权重不是自然部署频率。当前只有增量成本，shared 前缀不冒称 c0。单臂 DR、约束 bandit、完整验证 rollout 和冻结部署仍待后续卡。

## 2026-11-16 — P3-02 独立校准与选型数据入口

一名实现、两名独立只读审核。配对与在线单臂均支持显式 validation 子角色；默认训练入口拒绝验证数据，test 在采集/读取结局之前被拒绝；旧训练产物不改标。

| 文件 | 本卡变更 |
| --- | --- |
| `target_data/partitions.py` | 新增角色、子角色、任务子集、重入与来源库前置门禁 |
| `target_data/collection.py` | 准备记录绑定验证子角色，提前验证来源库 |
| `interventions/inputs.py` | 输入加载按冻结分区核对任务与轨迹 |
| `interventions/single.py` | 在线单臂按显式分区采样并冻结角色 |
| `interventions/collection.py` | 配对计划继承验证分区元数据 |
| `interventions/outcomes.py`、`single_outcomes.py` | 读取前拒绝错误分区，汇总保持验证绑定 |
| `policy/dataset.py` | 每个导出只接收一个显式分区；保留 train v1，validation 使用 v2 |
| `cli.py` | 采集和导出增加 validation-role，缺少子集时提前失败 |
| `tests/test_validation_partitions.py` | 新增 13 项分区、双路线、CLI、未知、泄漏与重入测试 |
| `tests/test_single_execution.py` | 内部测试 fixture 显式添加原有 target_train 角色 |
| `README.md`、`docs/task_cards.md`、`CHANGELOG.md` | 命令、验收和下一卡记录 |

上述源码路径均在 `src/cave_mem_v3/` 下。部分分区编辑已发生在本卡哈希快照前，因此文件列表结合完整差异报告。工程独立审核 58 项针对性及追加 16 项回归通过；研究审核 27 项及两个验证子集的 pending 探针通过。主 agent 首次全量发现三个旧测试 helper 缺 role，保留失败日志并修正 helper，最终 `python scripts/run_tests.py` **249 项通过（36.943 秒）**。旧 P3-01 两份最终 v1 数据集重新加载与原记录完全一致，`git diff --check` 通过。四条验证 CLI 路线仅使用 fixture；没有真实请求、训练或策略选型。

## 2026-11-16 — P3-01 第三部分只读数据入口

在分支 `codex/phase3-goal-workflow`、基线 `e3a6ef891f47a1cfd779eb710ed9bf0fd46bd398` 上按用户新要求使用一名实现 agent、两名独立只读审核 agent。先更新逐卡连续执行规则，然后只实施 P3-01，验收后才定义 P3-02。

| 文件 | 本卡变更 |
| --- | --- |
| `src/cave_mem_v3/policy/__init__.py` | 新增策略模块入口 |
| `src/cave_mem_v3/policy/dataset.py` | 从原始 prepared/paired/single 记录构建可复读的数据视图，隔离特征/标签/成本/来源；重核实际模型身份、实验条件、分组、重复与未知，独立标签掩码及费用有效性；独占导出，加载时重核原始证据 |
| `src/cave_mem_v3/cli.py` | 新增 `build-policy-dataset --paired RUN PREPARED --single RUN --output FILE` |
| `tests/test_policy_dataset.py` | 16 项数据入口、篡改、身份、分区、未知费用、分母和只读测试 |
| `AGENTS.md` | 持久化一写两审及验收后自动逐卡推进的最新授权 |
| `docs/software_goal.md` | proposal 与四张流程图到代码的依赖地图、完成判据和真实研究边界 |
| `docs/task_cards.md` | 完成 P3-01、记录验收与下一张卡 |
| `README.md` | 增加数据构建命令与原始来源要求 |
| `CHANGELOG.md` | 本条文件、验证与限制记录 |

验证：16 项针对性测试及 `python scripts/run_tests.py` 的 236 项全量测试通过，研究/工程两项独立审核通过，`git diff --check` 通过。既有 S2-01/S2-02 离线示例分别生成 `runs/p3-01-fixture-20261116/paired-dataset-final.json` 和 `single-dataset.json`，最终代码复读一致且原始文件字节不变；较早的 `paired-dataset.json` 因费用状态字段补全而被最终加载器拒绝，原文件保留未覆盖。两示例的来源库/manifest 不同，因此混合导出被正确拒绝。配对标签为 `D=.97, H=0, rescue=1`，单臂为 `Y=true, R=.96`，全部仍标记 fixture。准备失败费用为 `None/unavailable_not_validated`，附原始轨迹路径与哈希，未当作零。

限制：本卡仅接收现有 target_train 产物；保持原始 run 和任务资产可访问才能复读，不提供只有摘要的弱校验加载。不同配置、来源库和实际服务身份不静默混用；共享库下的结果不代表来源留出。没有训练、真实模型调用或正式数据采集。

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
