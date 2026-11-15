# 阶段二输入、决策与配对执行合同

版本：`phase2-contract.v1`，2026-11-15。对应任务：**V1-01**。

[任务卡](task_cards.md) · [V1 方案](v1_plan.md) · [研究设计](experiment_design.md)

## 范围与实现状态

本文件冻结后续实现需要遵循的数据、时间顺序、预算和错误语义。它不是已经可运行的 API，也不是实验结果。V1-01 只修改文档；没有实现下列新 schema、执行随机化、调用 actor/teacher、恢复真实容器或收集目标结局。

基线：commit `ee97ac12d4e43d4f168e6ce2acb5f2196c1a56f5` 加本地已有未提交内容，分支 `codex/v1-plan-20261115`。当前代码已生成 `target-run.v1`、`target-trace.v1`、`target-checkpoint-sample.v1`、`pre-treatment-state-record.v1`，但没有本文的六种下游合同。晚检查点预算问题仍待 V1-03 修复；新来源证据规则仍待 V1-02 实现。

V1 仅采集已审核文件系统协议内的单检查点配对数据，真实 pilot 限 `target_train`。未来学习策略 `π(z|x)`、单臂日志概率 `b(z|x)` 和配对执行顺序分别命名；V1 不训练 gate、不实现单臂 adapter、不运行 validation/test。Eq.2 使用小写 `e_t`，记忆中的大写 `E` 保留来源证据含义。

## 公共序列化与身份规则

- 下文表格标为“必填”的 key 必须存在；`T | null` 表示 key 必填、值可空。无值写 JSON `null`，禁止靠省略字段或空字符串表达不同状态。可选扩展必须在独立命名的版本化对象中声明，不静默接受新语义。
- `int` 不接受 bool；数值必须有限，禁止 NaN/Infinity。`Hash` 是 64 位小写十六进制 SHA-256。比例范围是 `[0,1]`；需要正概率时另行规定。
- 六种记录均必填 `schema_version` 与 `record_hash`。`record_hash = digest(整个对象去掉自身 record_hash)`，复用 `common/contracts.py` 的 `plain/canonical/digest`：UTF-8、`ensure_ascii=False`、键排序、分隔符 `(',', ':')`、`allow_nan=False`。数组顺序属于内容，不排序。
- 外部字段 `prepared_hash/eligibility_hash/plan_hash/arm_result_hash` 引用对应记录的 `record_hash`；原有 `run_hash/trace_hash/x_hash/a0_hash` 按各自现行 schema 重核。对象引用必须能定位原始文件和相应节点，不能只提供无法解析的孤立 hash。
- 六种新记录统一独占存储于本次干预运行根的 `records/<record_hash>.json`；外部 record hash 的 resolver 固定到该路径、根 JSON 对象，并重核文件内的 record_hash。创建下游记录前，上游记录必须已在该根目录可读；从其他 run 引入时只可原样复制并重核，不能修改内容。现有来源/准备文件继续使用下文 Ref；Ref.content_hash 是包含被引用对象自 hash 的整个对象 digest，与去掉 record_hash 后计算的记录 hash 不同。恢复/日志等其他原始产物不属于这六种记录，必须提供显式 Ref。
- 下游记录是新文件，不回写已冻结上游。新结局、日志完成状态和审核记录不得改变 `a0/x`。先写计划再执行；中断日志另写，不能通过修改原计划重新抽签或重发请求。
- 校验 hash 只证明内容一致，不证明任务范围已经审核、远端模型权重真实身份或随机流完全耦合。审批/审核对象、资产和实际恢复检查必须另有证据。
- 示例 JSON 是注明场景的**字段片段**，省略了公共身份字段；其中 `H_*` 是符号引用而非合法 Hash。它们只展示语义，不是可加载的真实运行文件、已实现的校验器输出或 benchmark 观测。

本合同采用单向依赖：`PreparedInput → EligibilityRecord → PairPlan → ArmResult → PairResult`。未来 `PolicyOutput` 只引用同一个 `PreparedInput`，不成为 V1 配对采集的资格门槛；不允许结果反向参与任一上游身份计算。

## A. 当前输入的身份与信息边界

以下是 V1-01 冻结的**拟实现合同**；现有代码只产出下表列出的输入，尚未实现本节三个新 record。具体类型名不表示已有 Python 类。所有记录是不可覆盖的 JSON 对象。

### A.1 编码与引用

- `H` 表示 64 位小写十六进制 SHA-256；`N` 为非负整数，拒绝 bool 冒充整数；`F` 为有限数，禁止 NaN/Infinity；`S` 为非空字符串。`T?` 表示必填字段但允许 JSON `null`，不是省略字段。
- 新 record 均必填 `schema_version:S`、`record_hash:H`，遵循公共序列化规则；不得通过重新排序证据数组“修复”旧 hash。
- 已有 `run_hash/trace_hash/x_hash/a0_hash/bank_hash` 按各自已有 schema 重核，不能全部按新 record 规则重命名；外层 trace 与内部 episode 的 `trace_hash` 是不同对象的 hash。下游的 `prepared_hash/eligibility_hash/policy_output_hash` 引用相应新 record 的 `record_hash`。
- `Ref={path:S, json_pointer:string, content_hash:H}`：path 相对本次运行根目录，不含密钥；pointer 遵循 JSON Pointer，定位该文件中对象，整个文件的 pointer 为 `""`。`content_hash` 是被引用对象的 canonical digest，若该对象本身带自 hash，则该字段也在被引用对象中。它与记录自身的 `record_hash` 不是同一算法输入，不能互换。
- 新 schema 中未列字段应拒绝；本合同明确允许的 `context/action/identity` 等不透明对象按其绑定版本校验。缺少必填字段、非法 enum、非有限数、重复 JSON key、hash/版本错配均拒绝，不以默认值补齐。
- 本文 JSON 示例是 **字段片段**，用于解释状态和条件，省略字段不表示完整 record 可省略；示例文字 hash 不是可运行证据。

### A.2 当前存储到合同的映射

| 当前产物 | 新合同读取位置 | 必须交叉核对 |
| --- | --- | --- |
| `target-run.v1` 的 `run.json` | `PreparedInput.run_ref/run_hash` | manifest、role、study、risk_hazard、actor/environment config、pre_treatment mode |
| `target-trace.v1` | `trace_ref/trace_hash` | trace→run、task→manifest、角色、seed、组信息 |
| `episode.checkpoint_sampling` | `sampling_ref` | status=selected、唯一选中 step、checkpoint hash、selected_action 与记录一致 |
| `episode.steps[i]` | `step_index/checkpoint_ref` | `action`、context、checkpoint 与 sampling 中的副本一致 |
| `step.pre_treatment_state` | `pre_treatment_record` 原样复制 | 内外 record/x/a0/context/retrieval hash 及 task/bank/memory 绑定 |
| bank + `audit_sources` | `bank_ref` 仅供审计器 | 全量来源证据按对应版本重核，原始审计轨迹不进入 gate/actor |
| manifest / 冻结运行清单 | `frozen_identity`、`dependency_groups` | 模板、已知依赖连通组、来源 lineage、资产/镜像/模型/规则身份 |

加载器验证整个输入链；它不重新生成动作、不再次检索或编码、不用结果重标 `ρ/e`。`collect-target` 的已完成轨迹不能因“找到了一个 step”而转换成准备输入；必须来自 `freeze_before_selected_action_execution.v1`，且选中时 episode 为 `pre_treatment_ready`、`success=null`、`utility=null`、无 private verifier、选中 step 的 result 为 null、未执行动作验证。

`input_integrity` 还核对冻结 actor 的 `capabilities.stateless=true`、其全部上下文由公开历史构造。V1 不支持私有会话缓存参与决策；否则记 blocked/`stateful_actor_unsupported`，不能仅靠环境快照声称恢复了 actor 状态。

校验仅要求 `a0` 是完整缓存的公开动作表示，符合缓存 schema；**不能在准入阶段调用 `env.validate_action(a0)`，也不能因预计动作不合法而排除样本**。动作执行是否合法属于两臂共同执行规则。缓存响应已在准备前解析失败且无合法准备 record 的情况，记 preparation failure，不伪造 `a0` 后进入配对。

无法解析/结构化缓存 `a0` 时记 `preparation_error`，进入准备阶段错误分母。必须检查原 `actor_response.parse_error`；当前 provider 的 `__unparsed_action__` sentinel 虽然是 JSON 对象，也不能当作成功缓存的动作。V1 的配对估计对象明确条件于“成功缓存原动作且满足技术资格的检查点”，不声称无偏覆盖所有原始任务。两臂开始后的 Remind 格式错误属于干预后行为，不得倒退改成 preparation_error 或从配对尝试分母剔除。

无选择轨迹可能按既有 prepare 流程自然结束并保留终局结果；此时只读取其选点分母和元数据，不能复制 episode 成功/效用到下游输入或资格规则。

## B. PreparedInput：经过校验的准备输入封装

Schema：`target-prepared-input.v1`。只有存在完整选中状态才能创建；无选择/输入不可读直接生成后述 EligibilityRecord，不伪造 PreparedInput。

| 字段 | 类型/可空 | 约束 |
| --- | --- | --- |
| `schema_version`, `record_hash` | S, H | 公共字段 |
| `contract_version` | S | 固定 `phase2-contract.v1` |
| `run_ref`, `trace_ref`, `sampling_ref`, `checkpoint_ref`, `bank_ref`, `manifest_ref` | Ref | 全部必填；checkpoint 是执行器私有输入 |
| `run_hash`, `trace_hash`, `episode_hash`, `manifest_hash` | H | 独立存放已有对象自 hash，禁止混淆内外 trace |
| `task_id`, `trace_id` | S | 与 run/trace/manifest 一致，只作审计键 |
| `role` | S | V1 固定 `target_train`；当前旧 CLI 允许 validation 不代表 V1 pilot 放行 |
| `validation_role` | null | V1 必须为空 |
| `dependency_groups` | object | `task_group:S`, `template_id:S?`, `known_groups:[S]`, `source_lineage:[S]`；来自冻结 manifest/来源库，不从结果推断 |
| `step_index` | N | 与唯一选中节点一致；不是整个 task 的新复本编号 |
| `checkpoint_hash`, `public_context_hash` | H | 与 selected step 及 sampling 副本一致 |
| `pre_treatment_record` | object | 原 `pre-treatment-state-record.v1` 完整内容，保留自己的 schema/hash |
| `sampling` | object | `selection_hazard:F`、`survival_probability_before:F`、`inclusion_probability:F`、`uniform_draw:F`、`sampler_seed:N`、`risk_hazard_hash:H` |
| `frozen_identity` | object | 见 B.1；审计元数据，不是模型特征 |
| `budget_audit` | object | 见 B.2；不修改 pre_treatment_record.x.b |
| `preparation_costs_ref` | Ref | 来源准备开销和请求账本；未计量值保留 null，不补零 |
| `run_manifest_ref` | Ref | 冻结运行清单，可解析 B.1 的身份和所有资产/提示/规则引用 |
| `randomization_status` | S | 固定 `not_assigned`；后续另建 PairPlan，不回写此字段 |
| `outcome_access` | S | 固定 `none`；目标结局不能参与构造或资格判断 |

sampling 的选中概率需满足 `0<hazard≤1`、`0<survival≤1`、`inclusion=hazard×survival`（浮点容差固定 `1e-12`），`0≤uniform_draw<hazard`。该概率条件于已观察到的无记忆轨迹前缀，不等于跨任务全局纳入率；不能单靠它外推所有未观测轨迹的自然分布。

`pre_treatment_record.status` 只允许 `prepared` 或 `prepared_no_memory`。前者要求 memory_id 非空、x.m 非 null、x.v=1；后者要求 memory_id/x.m 为 null、检索为 no_match、x.v/x.rho/x.e.value 保持 `?`。当前 top-1 非空已验证库不设置相似度拒绝阈值，不得在加载器新增阈值。

### B.1 frozen_identity 必填子字段

| 子字段 | 类型/约束 |
| --- | --- |
| `actor_config_hash`, `environment_config_hash`, `study_hash`, `retriever_config_hash` | H；逐一绑定当前 run 与冻结清单 |
| `benchmark_revision`, `task_assets_hash`, `base_image_identity`, `protocol_version` | S/H/S/S；实际资产与协议明确，不接受占位值 |
| `actor_prompt_hash`, `actor_weight_manifest_hash` | H, H?；网关无法认证权重时明确 null，并由部署清单声明身份限制 |
| `encoder_identity_hash`, `encoder_assets_hash` | H, H?；当前仅声明模型名/revision不等于权重 hash；真实 pilot 要求冻结本地资产清单，缺失则门禁 blocked |
| `exposure_rules_hash` | H?；无规则必须为 null，意味着 e 未知，不自动排除 |
| `feature_schema_version`, `structured_schema_version` | S, S?；前者取现有 `x.feature_map.status`（合格值为 `frozen_text_encoder_plus_structured.v1`），后者取 `x.feature_map.structured_schema_version`（当前为 `pre-treatment-structured.v1`）；缺失必需特征时不伪造版本，门禁 blocked |
| `source_rule_id`, `source_evidence_schema`, `source_bank_schema` | S?；有记忆时非空，无记忆为 null |
| `expected_response_identity`, `observed_draft_identity` | object；均为 `{model:S, fingerprint:S?, fingerprint_status:S}`，status 只允许 `reported/unknown`；模型别名须在 actor config 预登记 |
| `run_manifest_hash` | H | 完整运行清单的内容 hash，包含源 teacher 身份与证据范围、提示、预算和版本来源；不得含凭证 |

未知远端 fingerprint 本身不等于身份漂移；将已知型号/预期指纹不匹配视作阻断，实际无指纹如实声明耦合限制，不声称远端权重/RNG 已认证。发生 drift 不更新 expected 字段“吸收”变化。

### B.2 budget_audit 与未知特征

`budget_audit` 必填：`rule_id="remaining-budget-after-draft.v1"`、`status:known/unknown/mismatch`、`study_max_model_calls:N?`、`model_calls_consumed:N?`、`study_max_steps:N?`、`executed_steps:N?`、`environment_remaining_steps:N?`、`expected_model_calls_remaining:N?`、`expected_environment_steps_remaining:N?`、`matches_frozen_state:bool?`、`reason:S?`。已消费模型调用包括 `a0` 那一次；已执行步数不包括尚未验证/执行的 `a0`。这是一份审核结果，PreparedInput 的身份/hash 已核验不代表预算资格已通过。

重核 `R=study_max_model_calls−model_calls_consumed`、`L=min(study_max_steps−executed_steps, environment_remaining_steps)`，要求各差值非负，且 x.b 的 model_calls/environment_steps 分别等于 R/L。当前晚节点重复扣减缺陷由 V1-03 修复；已有错配输入保留原 x/hash，门禁 blocked=`budget_mismatch`，不能悄悄换一个 b 再沿用原实验身份。只有另行明确的迁移/新准备流程才可产生新记录。

status=known 要求所有计数可重核、matches=true、reason=null；status=unknown 将无法核定的计数及其依赖的期望值保留 null，matches=null、reason 非空；status=mismatch 保留原值和可算的非负期望值、matches=false、reason 非空。负差值不得钳制成零以通过验收，记 mismatch 并令该期望值 null。状态 known 仍可能因 R/L 太小被 excluded；unknown/mismatch 一律 blocked。

未知不是零：`x.rho`、`x.e.value` 保持 `0/1/"?"`；`x.e.confidence` 保留声明的低/中/高或 `?`，不是经校准概率；`x.d.value` 是有限 [0,2] 数或 `?`。未来 gate 的固定特征编码使用独立 unknown 类/缺失掩码；数值运算若需要占位 0，必须同时有 `observed=false`，不能把它当观测值。该编码规则按 feature_schema_version 冻结，不能训练后回写 x。

`ρ/e/d` 有未知值并不等于特征合同不完整。`feature_contract_incomplete` 指缺少整个必需 embedding/结构 schema、非法维度/非有限值或版本不明。V1 不训练 gate，但依提案仍要求冻结 encoder+structured 特征；明确声明的单项未知允许入组，不按 `ρ=e=1` 筛样本。

## C. PolicyOutput：策略建议与执行分离

Schema：`target-policy-output.v1`。所有字段必填；V1 仅建立“未训练”的占位记录，不执行 gate。它不决定是否采集配对，也不输出工具动作。

| 字段 | 类型/可空 | 条件约束 |
| --- | --- | --- |
| `schema_version`, `record_hash` | S, H | 公共字段 |
| `prepared_hash`, `x_hash` | H | 引用准确的输入与原 x |
| `policy_status` | S | `not_trained/scored/abstained`；V1 采集固定 not_trained，其余为未来部署接口 |
| `policy_id`, `policy_version`, `policy_weights_hash`, `calibration_hash` | S?, S?, H?, H? | not_trained 全 null；scored/abstained 必须有冻结模型及校准来源 |
| `feature_schema_version` | S | 与输入特征规则一致 |
| `pi_silent`, `pi_remind` | F? | not_trained 均 null；scored 有限 [0,1] 且和为 1（容差 1e-12）；abstained 固定 1/0 |
| `recommended_action` | S? | not_trained 为 null；scored 可为 silent/remind 或 null（若只输出分布）；abstained=silent |
| `tau_hat`, `harm_hat`, `incremental_cost_hat` | F? | not_trained 全 null；harm 若有须 [0,1]；收益/增量成本允许负数；缺失不填 0 |
| `reason_codes` | [S] | not_trained=[policy_not_trained]；abstained 需具体原因；scored 可空 |
| `assignment_status`, `outcome_access` | S, S | 固定 `not_assigned`、`none` |

未来若在 `recommended_action` 使用阈值/平局规则，其版本必须属于冻结 policy config，不通过简单 argmax 隐含新增决策规则。`PolicyOutput` 的 π 与单臂日志行为概率 `b(z|x)` 不同；配对的 `order_probability` 又是不同随机量。V1 没有 AssignmentRecord，也没有用 not_trained 替代成 `(0.5,0.5)` 的“假策略”。

只将经过允许列表投影的 `x` 交给 gate：context、a0、m 的公开字段、允许的来源证据 E 摘要、v/ρ/e/d/b、冻结特征。task/trace/group/source ID、原始 snapshot、audit_sources、验证器材料、目标结果、执行器私有路径只在审计封装内存在。hash/版本用于一致性检查，不能变成学习特征或用作按来源背答案的键。

## D. EligibilityRecord：配对技术资格，不预测提醒收益

Schema：`target-eligibility.v1`。无选择、无记忆及各种失败也须各保留一条资格/处置记录，使批次分母可追踪。

| 字段 | 类型/可空 | 条件约束 |
| --- | --- | --- |
| `schema_version`, `record_hash` | S, H | 公共字段 |
| `origin` | object | `{trace_path:S, raw_file_sha256:H?, trace_hash:H?, run_hash:H?}`；后两者只在重核后填写；文件不可读时 raw_file_sha256=null |
| `prepared_hash`, `x_hash` | H?, H? | 有已校验 PreparedInput 则非空；no_selection/输入无效则均 null |
| `status` | S | `eligible/excluded/blocked` |
| `reason_codes` | [S] | eligible 必须空；其余非空，去重后按 ASCII 字典序排列，不只保留首个原因 |
| `checks` | object | 固定键 `input_integrity/role/source_rule/features/budget/task_protocol/checkpoint_restore`，每个值为下述 Check |
| `expected_budget` | object? | 无 PreparedInput 时 null；否则 `{model_calls:N?, environment_steps:N?}`，复制 budget_audit 的期望值；eligible 时必须分别是已核定 R≥1、L≥1 |
| `restoration_audit_ref`, `checkpoint_probe_ref` | Ref? | eligible 必须非空；分别任务级范围审核与实际目标检查点检查，不能互相替代 |
| `policy_status` | S | V1 固定 `not_trained`，不要求 policy 成功预测 |
| `decision_stage`, `outcome_access` | S, S | 固定 `before_pair_plan`、`none` |

`Check={status:S, reason:S?, evidence_ref:Ref?}`；status=`pass/fail/unknown/not_run`，pass 的 reason=null，其余需原因。所有检查先执行无外部副作用的身份/角色/来源/预算检查，再在通过的对象上执行恢复探针；不对无记忆或无预算样本做无用的环境恢复。排除/阻断原因与未执行检查同时保留。

| status / reason | 进入条件 | 允许的下一步 |
| --- | --- | --- |
| eligible / [] | 完整输入、V1角色、新来源规则v=1、合格特征合同、已知 R≥1/L≥1、任务审核和实际检查点恢复全部 pass | 原子创建 PairPlan；后续资产漂移仍可阻断执行 |
| excluded / no_selection | 选点记录完整、没有选中节点；无 PreparedInput | 留分母后结束，无 PairPlan、无 PolicyOutput |
| excluded / no_memory | prepared_no_memory / top-1 no_match | 留无候选分母后结束，不制造空提醒 |
| excluded / insufficient_budget | budget 已知且一致，但 R<1 或 L<1 | 结束；不借预算、不执行单臂 |
| blocked / preparation_error、sampling_incomplete | 选点/准备调用中断、选中后 a0 无法解析或构造状态失败，没有完整 prepared record | prepared_hash/x_hash 为 null；保留准备错误分母与原始尝试，不生成 PolicyOutput/PairPlan |
| blocked / invalid_input、identity_mismatch、unsupported_role | schema/hash/时间边界/身份/角色失败或缺失 | 留拒绝证据；不重生成 a0、不运行两臂 |
| blocked / source_rule_mismatch、source_unverified | 旧/未知来源规则、v不是1、来源重核失败 | 结束；V1-02/18 另行产出新版本证据，不迁移旧标签 |
| blocked / feature_contract_incomplete | 必需 encoder/结构版本或 embedding 无效；不是单项ρ/e未知 | 修复配置后只允许新记录/新run，不覆盖旧 x |
| blocked / unknown_budget、budget_mismatch | 无法核定剩余额度，或冻结值与账本矛盾 | 保留原 x/hash；V1-03等后续卡修复，当前不继续 |
| blocked / restoration_unsupported、restoration_failed | 任务未审核/超协议、实际checkpoint恢复/隔离未通过 | 留原因，不自动改单臂，不降低审核标准 |

判定优先级：身份/完整性失败只能 blocked；信息完整后，no_selection/no_memory/已知预算不足为 excluded，其未运行检查记 not_run；有资格继续却因未认证或缺数据无法确认时 blocked。`eligible` 必须所有 Check=pass，不能用 unknown 或 not_run 凑齐。

新来源门禁固定规则 ID 为 `k2-all-corrections-success-positive-mean.v1`，精确条件为 K=2、两组 original/corrected 均完整有效、两次 corrected terminal success=true、平均效用差>epsilon。该名称是新协议 ID，不表示已实现 evidence schema；V1-02 负责实际 schema 版本及读写。旧 `source-evidence.v5`/旧 bank 仍按原规则可审计，不能只改 schema 字符串或看到 v=1 就混入新 pilot。source 旧结果不追溯改标；此门禁也不要求任意 95% 置信阈值。

## E. 输入与资格案例片段

以下均为文档例子，未运行任何实验；外层完整 hash/身份/引用按上述表补齐并核验后才可用于实现测试。

**正常输入，保留安全原动作组：** x.v=1、rho=1、e=0 不会被排除。

```json
{"pre_treatment_status":"prepared","x":{"v":1,"rho":1,"e":{"value":0,"confidence":"high","reason":"not_exposed_rule_matched"},"b":{"status":"known","remaining":{"model_calls":3,"environment_steps":4}}},"eligibility_status":"eligible","reason_codes":[]}
```

**未知暴露仍可配对：** 有完整 encoder/结构合同，公开条件缺证据只使值未知。

```json
{"x":{"rho":"?","e":{"value":"?","confidence":"?","reason":"no_frozen_action_mechanism_exposure_rubric"},"d":{"value":"?","reason":"source_checkpoint_context_unavailable"}},"eligibility_status":"eligible","reason_codes":[]}
```

**三种正常无输出：** 对应记录均保留分母；以下数组表示三条不同记录片段。

```json
[{"status":"excluded","reason_codes":["no_selection"],"prepared_hash":null,"x_hash":null},{"status":"excluded","reason_codes":["no_memory"],"pre_treatment_status":"prepared_no_memory","memory_id":null},{"status":"excluded","reason_codes":["insufficient_budget"],"remaining_model_calls":0,"remaining_environment_steps":2}]
```

**恢复阻断与坏预算：** task 审核失败不是 silent assignment；晚节点 budget 不能覆盖原 x 后通过。

```json
[{"status":"blocked","reason_codes":["restoration_unsupported"],"next_record":null},{"status":"blocked","reason_codes":["budget_mismatch"],"study_max_steps":4,"executed_steps":1,"environment_remaining_steps":3,"frozen_environment_steps":2,"expected_environment_steps":3}]
```

**未训练策略：** 缺失预测不是零收益，也不是随机分配概率。

```json
{"policy_status":"not_trained","pi_silent":null,"pi_remind":null,"recommended_action":null,"tau_hat":null,"harm_hat":null,"incremental_cost_hat":null,"reason_codes":["policy_not_trained"],"assignment_status":"not_assigned","outcome_access":"none"}
```

**选中后准备失败：** 原响应未能形成可缓存动作，只保留准备分母；不伪造 a0，也不能把后续 Remind 失败归到此类。

```json
{"status":"blocked","reason_codes":["preparation_error"],"prepared_hash":null,"x_hash":null,"expected_budget":null,"outcome_access":"none"}
```

**非法输入负例（必须拒绝）：** `not_assigned` 不足以证明输入无目标结果，不能忽略不利字段再通过。

```json
{"episode_status":"completed","success":true,"pre_treatment_status":"prepared","randomization_status":"not_assigned","expected_eligibility":{"status":"blocked","reason_codes":["invalid_input"]}}
```

## F. PairPlan：执行前冻结的配对计划

本节六类合同共用前文的 canonical JSON 与 `record_hash` 规则。下列类型是目标协议的规范，尚未由源码实现；现有来源回放、评分和历史产物不因此改变。

`schema_version="target-pair-plan.v1"`。仅 `EligibilityRecord.status="eligible"` 可以生成计划；顺序抽签之后不再变更其资格。必须先完成任务级范围审核和该实际检查点在原环境关闭后的恢复/文件隔离诊断，二者证据均绑定输入。每臂真正运行前仍要从原检查点恢复并重核初始状态，不能从诊断探针留下的分支或上一臂末态开始。

| 字段 | 类型/可空 | 约束 |
| --- | --- | --- |
| `schema_version`、`record_hash` | string，必填 | 固定版本；hash 校验遵循公共规则 |
| `pair_id`、`run_config_hash` | string，必填 | `pair_id=digest(["target-pair.v1",prepared_hash,replicate_index,run_config_hash])`；无结果参与 |
| `prepared_hash`、`eligibility_hash` | hash，必填 | 分别引用两类记录的 `record_hash`，且 eligibility 必须引用同一 prepared |
| `replicate_index` | integer，必填，≥0 | 复本属于同一检查点聚类；本合同不指定复本数，批次预登记 |
| `status`、`mode` | string，必填 | 分别固定 `frozen`、`paired`；执行状态另写 journal |
| `bindings` | object，必填 | 包含 `manifest_hash/task_id/trace_hash/checkpoint_hash/a0_hash/x_hash/bank_hash/memory_id/source_rule_id/source_evidence_hash/actor_config_hash/encoder_identity_hash/feature_map_hash/sampling_policy_hash/exposure_rules_hash/task_protocol_hash/task_audit_hash/checkpoint_audit_hash`；必须与 prepared、eligibility 和冻结运行清单一致；可缺暴露规则时该 hash 为 null，并继承输入中的未知原因 |
| `reminder` | object，必填 | `template_version/template_hash/text_ref/text_hash/max_chars` 均必填；提示正文在计划前渲染，文本长度不超过冻结整数上限；引用只包含允许的公开来源内容 |
| `order` | object，必填 | `arms` 只能为 `["silent","remind"]` 或反序；`probability=0.5`、`draw`∈[0,1)、`seed` 和 `rng_id` 必填；draw<0.5 为 Silent-first，否则 Remind-first |
| `seed_protocol` | object，必填 | 见下文；固定 stage 派生算法、实际 seed、请求/生效限制，不用文本“同 seed”代替记录 |
| `budget` | object，必填 | 见预算表；非负整数不能用 boolean，`R≥1`、`L≥1` |
| `cost_protocol` | object，必填 | 固定嵌套字段和单位见 F.1；权重、系数有限且非负，活动成本集合由系数/权重推导 |
| `execution_protocol` | object，必填 | 固定字段：`version="target-execution.v1"`、`actor_output_failure="stop_and_grade_current_files.v1"`、`finalization="original_private_verifier.v1"`、`other_arm_on_error="continue_only_after_local_error"`、`retry="no_automatic_retry"` |
| `policy_status` | string，必填 | V1 固定 `not_trained`；本计划不要求 PolicyOutput 给出预测 |

`bindings` 中的版本、配置内容和资产清单均须能通过其引用取回并重核，不能只有不可解析的名称。需要未知身份时以已审核声明和明确局限表达，不能编造 fingerprint 或权重 hash。`source_rule_id` 必须指向用户确认的新 K=2 规则，不能把 v5 的均值门槛当成同一规则。

计划不含 `assignment_probability`、`behavior_probability` 或选中的处理 `z`；配对执行两臂，0.5 仅为顺序概率。未来单臂协议需要独立 AssignmentRecord 和 `b(z|x)`，不能复用此字段充当 propensity。

### F.1 计量类型与冻结成本协议

`cost_protocol` 恰含以下必填字段：`version="target-cost.v1"`、`inference_weights:map<string,F≥0>`、`tool_weights:map<string,F≥0>`、`lambda_inference:F≥0`、`lambda_tool:F≥0`、`units:map<string,S>`、`active_cost_fields:[S]`、`operational_in_utility=false`、`unknown_policy="retain_null_and_disable_only_dependent_utility"`。两种 weights 非空，键必须属于 units；它们与冻结 StudySpec 相同。active_cost_fields 是满足 `lambda_inference*inference_weight>0` 或 `lambda_tool*tool_weight>0` 的维度并集，去重并按 ASCII 排序，不是手填筛选器。

units 的固定维度为 `model_calls/physical_attempts/tool_calls`（`count`）、`input_tokens/output_tokens/cached_input_tokens`（`token`）、`wall_seconds`（`second`）、`api_cost`（明确币种如 `USD`，未冻结币种则为 `unknown_currency`）、`tool_cost`（冻结 StudySpec 声明的非货币工具成本单位）。未知币种不能赋非零货币权重，缺报价仍记 api_cost=null。新增计量维度须升级成本协议，不接受临时字段。

`MetricMap={values:map<string,F≥0|null>, missing_reasons:map<string,S>}`：values 恰好覆盖 units 的所有键；missing_reasons 的键恰好等于 values 为 null 的键。count/token 维度必须是整数数值。没有实际发生且可证实的计量写0，发生但未知写null；null参与加法仍是null。MetricMap 使用其计划中的同一 units，不在结果里改单位。

原始事件/引用项类型 `ArtifactRef={id:S, kind:S, ref:Ref}`，kind 仅可为 `actor_attempt/tool_step/verifier_attempt/journal_event`。id 在同一 plan 内唯一；Ref.content_hash 验证完整原始对象，不额外用一个未定义的 hash 字段冒充记录身份。模型请求事件至少包含 `request_id:S, stage:S, requested_seed:N, applied_seed:N|null, status:started|response_completed|failed, usage:MetricMap, response_ref:Ref|null, error_code:S|null`；终止/响应事件不覆盖 started 事件，二者通过 request_id 关联。stage 是 remind_draft 或 continuation；每个物理尝试有独立ID，绝不自动重发。

### 种子与耦合

`seed_protocol` 必填 `version/base_seed/order_seed/remind_seed/continuation_actor_seeds/environment_seeds/provider_coupling/environment_coupling`。固定版本为 `matched-stage-index.v1`；使用当前 `stage_seed` 的 SHA-256 派生规则，以 `pair_id` 为 trial_id、`replicate_index` 为 replicate，stage 分别为 `pair-order`、`remind-draft`、`paired-continuation`、`paired-environment`。

`continuation_actor_seeds` 长度为 `R−1`，第 j 次普通续跑调用在两臂使用同一个 j 对应 seed；Remind 重提调用只消耗独立 `remind_seed`，不挤占该序列。`environment_seeds` 长度为 L，第 j 次实际工具执行对应同一索引 seed。未消费的槽保留为未用，不移动、补抽或返还。order 的随机数生成器及其实现版本在 `rng_id` 与运行清单冻结，日志保存实际 draw。

这是按阶段/调用序号声明的随机耦合，不是两个语义不同动作共享全部随机数的证明。provider 每次调用记录 requested/applied seed、model/fingerprint 和其实际能力；当前远端服务的权重、采样流及 Terminal 的未来 OS 时钟/RNG 不认证。`provider_coupling="requested_seed_only"`、`environment_coupling="filesystem_restored_future_rng_unmatched"` 是允许的明确局限，不得写成 exact RNG restoration。

### 预算

设 R 为缓存 `a0` 后、固定总模型调用上限内的剩余调用数；本节预算表中的 L 为当前动作前实际剩余环境步数（即 EligibilityRecord.expected_budget.environment_steps）。两臂各自遵守同一个部署预算，不把实验运行两臂的总开销误认为一次部署预算。

| `budget` 字段 | 值及含义 |
| --- | --- |
| `remaining_model_calls` | R；已扣除生成 `a0` 的调用，不再扣一次 |
| `remaining_environment_steps` | L=`min(study_max_steps−executed_steps, context.remaining_steps)` |
| `intervention_reserved_calls` | 1；Remind 重提一次使用，Silent 不使用也不转为续跑 |
| `continuation_model_calls_per_arm` | R−1；两臂相同，R=1 时允许为0 |
| `environment_steps_per_arm` | L；包含所选首动作；实际执行一次才扣一次 |
| `cached_draft_calls` | 1；仅用于共同成本说明，已经消耗，不是新增额度 |

R/L 未知为 blocked；已知且一致但 R<1 或 L<1 为 excluded，两者都不生成计划。选中后的动作不提前验证来提高资格。L 的既有重复扣减错误由 V1-03 修复，加载器不得修改已冻结旧 `x/b` 再沿用旧 hash。无效动作本身不消耗一次已执行工具步，但模型生成、API物理尝试和实际计费必须保留。诊断、恢复和 verifier 运维调用单列，不冒充 actor 工具步或无限制的免费动作。

## G. ArmResult：一臂的终止记录

`schema_version="target-arm-result.v1"`。每个 `(plan_hash,arm)` 最多一条不可变终止记录，另有可追加的阶段事件 journal；`record_hash` 不包括另一个分支的结果，避免循环绑定。

| 字段 | 类型/可空 | 约束 |
| --- | --- | --- |
| `schema_version`、`record_hash`、`plan_hash`、`pair_id` | string/hash，必填 | plan_hash 引用 PairPlan.record_hash，pair_id 必须一致 |
| `arm` | enum，必填 | `silent` 或 `remind` |
| `status` | enum，必填 | `completed`、`infrastructure_error`、`blocked`；未确认外部请求完成则只保留 pending journal，不制造终止记录 |
| `termination_reason` | enum，必填 | `task_terminal`、`budget_exhausted`、`invalid_actor_output`、`restore_failure`、`identity_drift`、`provider_failure`、`tool_backend_failure`、`verifier_failure`、`protocol_violation`、`shared_precondition_lost` |
| `restoration` | object，必填 | `expected_checkpoint_hash/restored_checkpoint_hash/state_fingerprint/audit_hash/status`；恢复前错误时实际 hash/fingerprint 为 null；执行前必须 matched |
| `initial_action` | object/null，必填 | `action/hash/origin/validation_status/reason`；Silent origin=`cached_a0`且字节等价 canonical/action hash匹配；Remind origin=`remind_redraft`；没有可解析动作时 action/hash可null，保留响应引用 |
| `reminder` | object/null，必填 | Silent为null；Remind包含 `template_hash/text_hash/response_ref/response_hash`；尚未收到响应时响应字段为null |
| `actor_behavior` | enum，必填 | `valid`、`invalid_action`、`malformed_action`、`not_observed`；它不是 terminal_success 的别名 |
| `attempt_refs`、`step_refs` | [ArtifactRef]，必填 | 可空；前者仅actor_attempt/verifier_attempt，后者仅tool_step；响应raw text仅作审计，不交给另一臂 |
| `response_identity` | object，必填 | 固定结构见 G.1；无模型调用时observed=null，不伪造第二份模型响应 |
| `costs` | object，必填 | 恰含 `intervention:MetricMap/continuation:MetricMap/finalization_operations:MetricMap`；最后一组包括本臂恢复和评分运维；单位由计划冻结 |
| `terminal_success` | bool/null，必填 | 仅原私有 verifier 完成并给出明确结果时为bool；未评分、未完成或infra错误为null |
| `verifier` | object，必填 | `status/definition_hash/result_ref/result_hash`；status=`completed/not_run/error`；非completed引用可null，不能伪造测试通过/失败 |
| `utility` | number/null，必填 | 从 terminal_success 和冻结成本口径重算，不信任调用方自填值；缺活动成本时null |
| `outcome_valid`、`utility_valid` | bool，必填 | outcome_valid须恢复、身份、协议均有效、无infra且verifier完成；utility_valid还须效用依赖的成本齐全 |
| `error` | object/null，必填 | 完成且无infra时null；否则 `class/scope/code/attempt_ref` 必填，已知缺失的attempt_ref可null，不写密钥/响应headers |

### G.1 响应身份与嵌套结果

`response_identity` 恰含 `requested:{model:S, revision:S, expected_fingerprint:S|null, config_hash:H}`、`observed:[Observation]|null`、`status:consistent|unknown|mismatch|not_observed`。`Observation={request_id:S, response_model:S|null, fingerprint:S|null, requested_seed:N, applied_seed:N|null, seed_support:applied_reported|requested_only|not_supported|unknown}`；按物理尝试顺序记录，成功响应必须有 response_model，失败且尚无响应时允许null。是否支持 seed 不能由“已发送”推断；null指纹不伪装为mismatch，已知不匹配不得被后续响应覆盖。

没有调用时observed=null/status=not_observed；有调用且任一已知身份不匹配时status=mismatch；其余若任一实际身份缺失则unknown，否则consistent。请求响应可仅声明 seed，而模型身份一致；seed能力仍单独按实际报告保存，不能由consistent推出RNG完全耦合。

`restoration.status` 为 `matched/failed/not_run`，其中 `expected_checkpoint_hash:H`、`audit_hash:H` 必填，其余实际hash为 H|null。`initial_action` 对象内 `action:object|null, hash:H|null, origin:cached_a0|remind_redraft, validation_status:not_run|accepted|rejected|malformed, reason:S|null` 全部必填；执行未开始时整个对象可null。`verifier.definition_hash:H` 必填，`result_ref:Ref|null` 与 `result_hash:H|null` 在completed时必填非空；result_hash取原评分产物schema声明的自hash，若无自hash则取Ref.content_hash。`error={class:S, scope:arm_local|pair_shared, code:S, attempt_ref:Ref|null}`，其代码必须能由被引用异常/日志核对。

### 两臂执行与错误规则

Silent 先校验并执行缓存 `a0`，该动作前 actor 调用数必须为0；Remind 先将已冻结提醒随同一公开上下文传入目标专用入口，恰好一次重提，然后校验并执行该动作。两臂后续都使用无提醒的普通续跑。来源 actor renderer 的禁提醒保护不解除。

入组条件要求已有结构可表示的原 `a0`；原草稿解析失败、只能得到 `__unparsed_action__` 占位的旧记录属于 preparation_error，无PairPlan，并计入准备失败分母。已结构化但违反动作权限/有效性规则的 `a0` 不在资格阶段提前剔除，留给Silent执行前验证。Remind是入组后的干预，其解析/动作失败不能倒退为准备失败排除。故本配对效应以“成功形成冻结原动作与状态”的检查点人群为条件，报告须公开这一覆盖限制。

| 发生的事件 | 两臂共同处理 | 结果与配对影响 |
| --- | --- | --- |
| 正常终局或预算耗尽 | 停止actor，使用原私有verifier评当前文件；不为评分再采一次动作 | verifier明确bool且其余合同有效才有Y |
| 已完成actor响应但JSON/动作格式不合法，或动作被权限/有效性规则拒绝 | 保留响应与费用，**停止actor，不执行该动作**；原私有verifier评当前合法文件；无补采、无动作替换 | actor_behavior记invalid/malformed；Y来自实际评分，可能true或false；不可仅排除Remind的此类样本 |
| shell命令返回非零，但工具执行和观察正常 | 当作公开任务反馈，按冻结actor正常续跑 | 不因不利命令结果标基础设施错误 |
| 网络/调用超时、provider封包无法可信读取、权限系统故障、工具后端或verifier故障 | 保留尝试/费用及可得日志，不自动重试；不能将无结果写成任务失败 | infrastructure_error；terminal_success/utility为null，不能形成完整配对标签 |
| 快照/资产/actor身份漂移、跨臂污染、冻结预算或顺序协议被破坏 | 停止整个pair，已完成部分仍保留；不重新恢复补救成另一pair | 共同前置条件失效，后臂blocked；原始Y可审计，但不得生成有效配对标签 |

上表对无效actor输出的终局处理是**目标协议的新约定**；现有 runtime 会直接给 `invalid_action/success=false`，目标实现必须在 V1-08/09 专门遵循上述原verifier口径，而不改写来源历史结果或将协议错误假称为官方测试失败。无效输出对应的ArmResult `status=completed`、`termination_reason=invalid_actor_output`，仅在评分和其他检查完成后 `outcome_valid=true`。

一个已明确结束、仅局限于当前分支的 provider/tool/verifier 故障允许记录第一臂失败后，重新核验共同前置条件，再按原计划执行另一臂；不能补跑失败臂。恢复失败、共享身份/资产/协议漂移或跨臂污染一律停止整对。若外部请求是否结束未知（pending），整个pair停止自动执行，不启动下一臂，待另一个明确的审计任务核对；本任务合同不提供自动和解或重复请求。

### 持久化和状态机

`PairPlan frozen → pair.started → arm.restore.started → arm.restore.checked → [remind.request.started → remind.response.completed] → first_action.checked → continuation → verifier.completed → arm.result → next_arm → pair.result`。

每个有副作用/费用的外部请求前先持久化唯一 request/attempt ID 与 `started`；完成响应后立即落盘，再执行下个动作。每次环境step、评分和分臂结束也须落盘；不能只在整对结束写一个JSON。第二臂只消费同一计划，不读取第一臂的提示/动作/观察/评分来改变行为。

命令重启时：已有可重核ArmResult只读取、不执行；只有started且无法确认结果的请求保持pending并停止；已完整记录的局部失败是失败终止记录，不自动“重试成功”。pair/arm journal由编排器负责，CLI只负责批次调度。写入使用独占创建；可重算summary不替代原始记录，原始记录和旧结局不可覆盖。

## H. PairResult：配对有效性、标签与覆盖

`schema_version="target-pair-result.v1"`。它是已有日志的派生快照，不执行模型或环境动作，不参与资格、顺序或提示的决定。

| 字段 | 类型/可空 | 约束 |
| --- | --- | --- |
| `schema_version`、`record_hash`、`plan_hash`、`pair_id` | string/hash，必填 | 与原计划一致 |
| `arms` | object，必填 | 固定键 `silent/remind`，值为ArmResult的record_hash或null；null必须有对应missing_reason |
| `status` | enum，必填 | `complete/partial/blocked/pending`；按下文唯一优先规则推导，不允许调用方自定 |
| `journal_refs` | [ArtifactRef]，必填 | kind仅journal_event；覆盖已发生的pair/arm started、完成、全局阻断和pending依据；独占原始事件，不引用易变summary |
| `missing_reasons` | object，必填 | 对缺失臂注明 `not_started/pending/shared_precondition_lost` 等可核查原因；完整时空对象 |
| `paired_outcomes_valid` | bool，必填 | 两臂outcome_valid、恢复和身份共同匹配、没有跨臂污染、顺序/预算合规；与成本缺失分开判断 |
| `paired_utility_valid` | bool，必填 | paired_outcomes_valid且两臂utility_valid |
| `labels` | object，必填 | `D/H/rescue` 均必填；D为finite number或null，H/rescue为0/1或null；条件见下文 |
| `cost_summary` | object，必填 | 恰含 `shared/arm_silent/arm_remind/operational/experiment_total/deployment_silent/deployment_remind`，每项均MetricMap；缺臂将未观测支出记null并说明原因 |
| `cluster_keys` | object，必填 | `task_id/checkpoint_hash/template_id/dependency_component/source_lineage`；来自冻结manifest/来源，不按结果重新分组；不存在template等须明确null及原因 |
| `invalid_reasons` | array，必填 | hash、身份、协议、infra、成本缺失原因可多项，不能只保留有利标签；无异常为空列表 |
| `evidence_scope` | string，必填 | 声明文件系统恢复与请求seed耦合范围，不声称独立个体因果效应或官方Harbor条件等价 |

状态按以下优先级从同一不可变日志快照推导，第一条满足即唯一状态：

1. 任一外部请求只有started、不能确认结束：`pending`；保留同时存在的共享故障原因，不自动继续。已有终止record与同臂仍pending矛盾时拒绝快照。
2. 无pending，但有 `pair_shared` 故障、pair.blocked事件或任一ArmResult.status=blocked：`blocked`，即使两个终止record都已存在也不记complete。
3. 无前两项且silent/remind两个终止record齐全：`complete`，允许其中一臂为已结束的arm_local infrastructure_error；此时仍不能生成有效配对标签。
4. 其他合法情况：`partial`，包括尚未开始或仅一臂终止、另一臂not_started；不得借partial自动补跑已终止失败臂。missing_reasons恰好覆盖arms为null的键。

PairResult 是带新record_hash的派生快照；后续原始事件到来可产生另一份结果快照，不能覆盖旧快照。complete描述日志结束，不等于有效；若shared协议失败之后发现，即使旧快照曾有效，也保留旧文件并用新的blocked审计快照声明无效，汇总须读取最新可重核事件视图。

合法标签：`H=1{Y_silent=true且Y_remind=false}`，`rescue=1{Y_silent=false且Y_remind=true}`，仅 `paired_outcomes_valid=true` 时计算。`D=U_remind−U_silent` 仅 `paired_utility_valid=true` 时计算。缺臂、恢复失败、infra错误或身份不一致时三者均为null；不能把缺失当0。

这里把“完整有效pair”细化为**结局有效**与**效用有效**：两臂真实Y已知而某个有权重成本缺失时，H/rescue仍可辨认，D保持null；不得因成本missing把已观察的回退藏掉。报告必须分别列出这两种有效pair的分母，不能用同一个n混算。

### 成本与效用的可重算口径

计量单位由冻结cost_protocol给出，例如model_calls、physical_attempts、input/output tokens、tool_calls、tool_cost、api_cost、wall_seconds；同一计量若未知用null，不因服务未报价就声称0货币成本。

`shared` 是干预前同一次目标前缀与缓存草稿/检索/特征构建的实际开销，只发生一次。`arm_z` 包含该臂干预额外调用与普通续跑；Silent干预调用为已知0，Remind为实际一次调用及费用。恢复诊断、实际恢复、verifier与实验编排的运维成本单列 `operational`，包括两臂finalization_operations且只汇总一次。V1固定 `operational_in_utility=false`：运维仅计实验实际支出，不进入U；将其纳入部署效用属于后续协议变更，必须另定成本归属和新run，不能事后调整。

对应单臂部署成本 `deployment_z=shared+arm_z`；实验实际支出 `experiment_total=shared+arm_silent+arm_remind+operational`。对可加的计数/费用逐维计算；wall time同时报告实际start/end elapsed，不把两臂墙钟简单相加冒充真实总耗时。原始分段wall_seconds仍保存。

`U_z=Y_z−lambda_inference*weighted_inference(deployment_z)−lambda_tool*weighted_tool(deployment_z)`，函数口径与冻结StudySpec一致；分支效用只从实际成本重算。共享成本在两臂差中抵消，但仍须公开报告单臂部署绝对成本与未干预的基线草稿开销。若共同成本缺失使任一U未知，V1保守令D未知，不事后凭抵消另造D估计；可在未来独立分析协议中处理。

仅**系数与该维权重乘积非零**的缺失维度令utility_valid=false；未加权的缺失api_cost仍记unknown但不妨碍例如只按调用次数计费的效用。不得删掉缺失维度或改变权重使本批数据可用。Y未知永远不能计算U。

无选择、无记忆、未通过资格的PreparedInput不创建PairPlan，因此不伪造PairResult；它们保留在输入/资格覆盖清单，汇总时与pair数量一同报告。每批至少报告计划rollout、无选择、选择后无记忆、资格阻断、计划pair、尝试/终止臂、完整pair、结局有效pair、效用有效pair、pending和各类错误；不能只输出有标签的训练行。

## I. 执行与结果示例（规范片段）

下面JSON是可解析的字段片段，省略公共身份、引用和hash，不是可运行的完整产物；符号hash只是说明位。所有数字为人为例子，不是真实benchmark结果。

### 正常预算与顺序

已缓存a0后R=4、L=3，Silent-first；两臂均最多3次普通续跑，Remind再用1次预留重提；两臂各最多3个实际工具动作。R=1、L=1也合法：普通续跑0次，分别执行缓存/重提的一个动作，随后评分当前文件。

```json
{
  "schema_version": "target-pair-plan.v1",
  "mode": "paired",
  "order": {"arms": ["silent", "remind"], "probability": 0.5, "draw": 0.2, "seed": 17, "rng_id": "frozen-example-rng"},
  "budget": {"remaining_model_calls": 4, "remaining_environment_steps": 3, "intervention_reserved_calls": 1, "continuation_model_calls_per_arm": 3, "environment_steps_per_arm": 3, "cached_draft_calls": 1},
  "policy_status": "not_trained"
}
```

### actor无效输出不是伪造的评分失败

```json
{
  "schema_version": "target-arm-result.v1",
  "arm": "remind",
  "status": "completed",
  "termination_reason": "invalid_actor_output",
  "actor_behavior": "malformed_action",
  "terminal_success": true,
  "verifier": {"status": "completed", "definition_hash": "H_VERIFIER", "result_ref": {"path": "private/verifier.json", "json_pointer": "", "content_hash": "H_RESULT_CONTENT"}, "result_hash": "H_RESULT"},
  "outcome_valid": true,
  "utility_valid": true,
  "error": null
}
```

这个例子说明当前文件可能已经满足原测试，因此畸形新动作不自动意味着Y=false；如果原测试失败则Y=false。两臂均按同一规则处理，仍保留invalid_actor_output行为指标。

### 正常回退与缺失成本

只惩罚model_calls，lambda=0.01：shared=2次，Silent分支3次、Remind分支4次；Y_silent=true、Y_remind=false。U_silent=0.95，U_remind=−0.06，D=−1.01，H=1，rescue=0；实验模型调用总数=9，单臂部署分别5、6。

```json
{
  "schema_version": "target-pair-result.v1",
  "status": "complete",
  "paired_outcomes_valid": true,
  "paired_utility_valid": true,
  "labels": {"D": -1.01, "H": 1, "rescue": 0},
  "invalid_reasons": []
}
```

若冻结权重还包含api_cost但Remind实际费用unknown，保留相同Y和H/rescue，改为 `paired_utility_valid=false`、`D=null`、`invalid_reasons=["weighted_cost_unknown:remind.api_cost"]`；不能临时去掉api_cost权重。

### 缺臂/错误不能成为零回退

```json
{
  "schema_version": "target-pair-result.v1",
  "status": "pending",
  "arms": {"silent": "H_SILENT_RECORD", "remind": null},
  "missing_reasons": {"remind": "pending"},
  "paired_outcomes_valid": false,
  "paired_utility_valid": false,
  "labels": {"D": null, "H": null, "rescue": null},
  "invalid_reasons": ["remind_request_completion_unknown"]
}
```

若Remind有明确infrastructure_error终止记录，则其hash不为null，PairResult.status可为complete，但两种validity仍false、全部标签null。已有Silent的观测仍保留，不能称为按已知b采集的独立单臂实验。

### 必须拒绝的组合

| 非法组合 | 拒绝原因 |
| --- | --- |
| 未通过实际检查点恢复门禁却生成PairPlan | 缺失配对资格；任务级审核不能代替逐点审核 |
| PairPlan含 `behavior_probability=0.5` 并称已单臂随机分配 | 混淆执行顺序与处理概率 |
| Silent在首动作前重新调用actor，或Remind有两次重提 | 破坏缓存原动作或一次提醒合同 |
| 两臂从不同checkpoint或后一臂读取前一臂结果 | 恢复/信息隔离无效 |
| R=4但Silent普通续跑上限4、Remind普通续跑上限3 | 未使用相同R−1续跑额度 |
| arm status=infrastructure_error但terminal_success=false | 把缺失结果伪装成任务失败 |
| actor malformed_action直接写terminal_success=false且无verifier | 未按原评分器获得结局 |
| 缺臂却H=0，或有权重费用未知却D是数值 | 无可识别标签或效用 |
| started未确认请求在重启后自动重发 | 破坏一次物理尝试与中断保留规则 |

## J. V1-01 验收索引与后续实现边界

| V1-01 要求 | 本合同位置/决定 | 实现任务 |
| --- | --- | --- |
| PreparedInput/PolicyOutput/EligibilityRecord | B/C/D 的必填、可空、hash、角色、资格与原因字段 | V1-04/05；学习策略只预留接口 |
| PairPlan/ArmResult/PairResult | F/G/H 的计划、分臂终止、结果及日志状态 | V1-06/08/09/10/11/12 |
| 正常与无选择/无记忆/未知特征/预算不足/恢复不支持 | D 的状态表、E 的场景片段 | 后续按对应卡写有意义的验收测试 |
| 缺臂/错误与标签条件 | G 的错误表、H 的状态优先级和双重有效性、I 的负例 | V1-10/11 |
| 一次a0、一次提醒与R−1预算 | A 的入组边界、B.2预算重核、F/G的执行规则 | V1-03/08/09 |
| 成本与结果分离 | F.1与H：冻结单位/权重；运维仅计实验支出；未知活动成本使D缺失，不隐藏已观测H | V1-11 |
| 新K=2与历史兼容 | D：新规则ID；旧v5只按旧规则审计，不改标或混入新pilot | V1-02/18 |
| 不可恢复与单臂边界 | D阻断；F顺序概率不等于单臂b | F-05另立协议，不在V1-01实现 |

编写分工：`contract_inputs_writer` 编写输入/策略/资格草稿，`contract_execution_writer` 编写计划/执行/结果草稿，主 agent 统一字段与引用；`contract_reviewer` 独立审核。所有示例是人为构造的文档片段；没有真实运行次数、模型调用费用或benchmark结局可报告。

本卡只冻结合同。V1-02负责新来源证据版本，V1-03负责实际预算修复；它们未因本文件完成而自动执行。真实任务、模型revision、资产hash、具体预算/成本权重/样本量仍须在V1-14冻结。本文件不填造这些运行值，也不将此前139项软件测试当作新合同已经实现的证明。
