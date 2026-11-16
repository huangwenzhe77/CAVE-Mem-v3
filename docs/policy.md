# 第三部分：训练接口与解释范围

本页记录逐卡实现的轻量软件基线。离线 fixture 验收不证明真实提醒效果，也不确定正式研究参数。当前卡状态见 [任务卡](task_cards.md)。

## 安装与数据

在独立虚拟环境中安装 `python -m pip install -e ".[policy]"`。此 extra 使用 NumPy；第一、二部分的基础命令无需此依赖。训练数据必须通过 `build-policy-dataset` 从原始日志重核；配对原始 prepared 目录与任务资产仍需保留。训练、validation/calibration、validation/selection 分别导出，不能临时按行随机拆分。

自动分流采集目录使用 `cave3 build-policy-dataset --auto <run-dir> --output <new-dataset.json>` 导出。每行仍标记 `paired` 或 `single`：前者可有真实配对 `D/H`，后者只含随机分配的一臂及其 propensity；未完成会话不能进入训练数据。

## 配对监督与校准

`train-policy` 只接受 target_train。冻结文本向量与明确的结构化字段构成轻量输入：v/ρ/e 使用三值编码，e 置信度使用分类编码，连续值附未知标记。当前已记录文本向量表示公开上下文和 a0，记忆通过 E、ρ/e、检索与距离字段进入；没有额外编码原始记忆全文。任务 ID、来源 ID、分配概率、目标成本与结局不进入特征。归一化只由训练数据计算。

收益 τ 与各增量成本分量使用带 L2 的线性回归；风险 h 使用带 L2 的 logistic BCE。各头按自己的有效标签掩码拟合，单臂不会构造 H，未知费用不补零。组内复本合计相同权重，记录组轴与实际样本数。D 已包括预登记成本，预测后不再扣一次成本。

`calibrate-policy` 只读取独立 validation/calibration。收益/成本使用组级最大绝对残差的 split-conformal 预测范围；这是未来实现值的预测区间，不是条件均值的置信区间。风险按预先固定预测分箱统计组内是否出现伤害并保留正的不确定性上界；它不是逐个 x 的安全保证。组数不足、缺头或无有效标签时保持 unavailable。默认参数只是软件默认值，正式 alpha、组数和阈值需预先登记。

`train-initial-gate` 在校准之后，仅用原训练集 x 与明确阈值生成监督目标，再训练同维度 logistic gate。不使用校准行训练 gate。校准不足、来源无效、收益下界不足或风险上界过大时目标为 Silent；单类别等不足训练状态也明确保留。后续 bandit 应保持相同输入与 gate 容量。

```bash
cave3 train-policy --dataset <train.json> --config <fit.json> --output <new-model.json>
cave3 calibrate-policy --model <model.json> --dataset <calibration.json> --config <calibrate.json> --output <new-calibration.json>
cave3 train-initial-gate --model <model.json> --calibration <calibration.json> --dataset <train.json> --config <gate.json> --output <new-gate.json>
cave3 predict-policy --model <model.json> --calibration <calibration.json> --gate <gate.json> --features <frozen-x.json>
```

以上 config 是明确的 JSON 参数记录；API 的 `SupervisedConfig`、`CalibrationConfig`、`GateConfig` 定义合法字段，配置随产物保存。模型和校准产物为版本化 JSON；不加载 pickle，不覆盖既有文件。预测不会调用 actor、teacher、环境或 encoder。

## 解释范围

`target_group` 按任务/模板/依赖分组，允许固定来源库共享，但只支持该库条件下的目标泛化。`source_target_group` 还检查实际来源关联，不能仅通过更换组 ID 隐藏共享来源。产物保存配置、分区、数据哈希、编码器及实际服务模型身份。

本阶段预测的成本仅为已审核的配对续跑成本差。原始 shared 包括任务前缀，不能冒称纯草稿/检索/gate 开销 c0。后续模块已接通单臂 DR、bandit、完整验证 rollout 和最终冻结评估；约束预算明确限于 `selected_gate_actor_calls`，不是完整任务成本约束。离线预测本身不执行动作，实际执行需使用 [冻结评估协议](evaluation.md)。

## 单臂交叉拟合与 DR

`estimate-single-dr` 只接收已审核 target_train 单臂数据。配置的 `candidate_remind_probability` 是此次诊断的固定候选提醒概率，必须显式提供；它与日志中随机分配概率 b、检查点纳入概率是三个不同量。

```bash
cave3 estimate-single-dr --dataset <single-train.json> --config <dr-config.json> --output <new-dr-report.json>
```

`DRConfig` 还记录 folds、seed、group_axis、minimum_arm_groups、l2 和 inverse_weight_caps。组按配置先分折，每折仅用其他组拟合归一化及 μ0/μ1。两臂训练支持不足时，该折对应记录不可估计，原有分母和失败原因仍保留；不为了得到可用值按结局重新分折。

每条有效记录构造 `ψz = μz(x) + 1{Z=z}/b(z|x) × (R−μz(x))`，候选策略价值为 `(1−p)ψ0+pψ1`，相对 Silent 的改进为 `p(ψ1−ψ0)`。ψ 是估计量，不是补出的配对观测；输出 estimated_dr_advantage 供后续训练使用，不生成已观测 H/rescue。未知效用、费用或结局保持缺失。

报告等组均值、有效行/组分母、原始权重和 ESS，并单列按配置截断后的估计及其相对原始估计的偏移。截断偏移是敏感性诊断，不是已知偏差界。极端权重的汇总溢出显式标记，不生成伪造有限值。等组均值不代表自然任务频率；完整正式任务指标还要包含未选中和无记忆的任务。

固定配置先于本次日志加载，不代表用户此前从未看过结局。利用这些数据训练或挑选的策略，其训练内 DR 值不是独立评估；最终策略仍需独立完整验证 rollout。配对与 DR 是两条单独估计路线，不能混称。

当前单臂 DR 统计可独立运行，但单臂 bandit 的初始化与冻结仍依赖兼容的配对 model/calibration/warm_gate。完整演示也先建立配对模型，再用单臂 DR 优化候选；这不等于只收集不可恢复单臂数据就能从零训练整套策略。完全无配对数据的初始化与校准方案需要另行定义，不能从单臂结果伪造风险监督。

缺失结果若与分配臂或潜在结果有关，完整案例选择可能破坏原始随机化；不能仅凭原日志 b 就将可用行 DR 解释为总体价值。报告保留缺失分母，真实解释还需要缺失机制假设。

## 约束 bandit 候选

`train-bandit` 在监督初始 gate 的相同特征、归一化与 logistic 参数维度上优化。配对路线使用 π(1|x)D；单臂路线使用独立保存的 OOF DR advantage，二者不合并成一种观测。目标扣除与 SFT 的 KL 正则，同时对经验组 bootstrap 风险/成本上界作非负对偶更新。每轮保存目标、KL、均值、上界和对偶变量，同容量无风险约束对照单独保存。

```bash
cave3 train-bandit --dataset <train.json> --model <model.json> --calibration <bounds.json> --warm-gate <gate.json> --config <bandit-config.json> --output <new-bandit.json>
# 单臂路线还需 --dr <audited-crossfit.json>
```

BanditConfig 必须明确 route、risk_limit、actor_call_budget；可记录步数、学习率、对偶学习率、KL、bootstrap 次数/尾概率、最少组数、种子等。输出只表示候选，不能自动部署或作为独立验证赢家。初始 gate 不可用、原始成本不充分、待完成记录或最终约束不满足时，明确不可用/不可行。

第一版预算单位为 selected_gate_actor_calls：c0 是从原始选中草稿响应验证的 1 次 actor 调用，本地检索/特征/gate 不请求 actor；配对 Δc 使用两臂已审核调用差，允许负差；单臂使用剩余调用预算作保守增量上界，不伪造另一臂实际调用数。它不约束 token、耗时、外部费用或完整任务调用数。共享前缀总费用仍不等于 c0。

配对风险使用已有校准范围，单臂没有已观测 H 时使用 h=1 最坏情况界。ρ/e 在主策略中保持软特征。特征存在但预处理资格不满足的样本提醒概率/梯度为零，仍参与组均值且计入已审计 c0；未选中和缺失状态保留独立分母，缺失不能悄悄制造可行候选。经验约束和有限次优化不提供总体安全保证，真实模型只能通过后续完整 validation rollout 选择。
