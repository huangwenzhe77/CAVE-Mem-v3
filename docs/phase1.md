# 第一部分：历史失败记忆构建

[返回项目首页](../README.md) · [项目架构](architecture.md) · [完整实验设计](experiment_design.md)

本文命令均从仓库根目录运行。配置已按 `configs/phase1/` 与 `configs/benchmarks/` 分类。

本指南说明 proposal §3.1：历史任务采集、失败修正假说、独立来源回放和冻结记忆库。第二部分的可恢复 Silent/Remind 软件闭环已实现，使用入口见 README；第三部分训练、第四部分部署尚未实现。

执行模型可以选择 **Qwen/Qwen2.5-7B-Instruct**。它承担冻结 actor 的角色，训练对象在后续阶段才是提醒控制器。当前 H20 服务器配置用于工程 pilot：每题最多 20 步/20 次 actor 调用，每个候选 K=2 组回放。教师使用未认证权重版本的网关别名。这些参数仅用于跑通流程；正式来源实验的任务、预算和证据标准仍须另行确定。读取配置或跑软件测试不会请求模型。

## 实验选择

### Benchmark 如何划分

在采集前固定一份任务清单，并按任务及已知依赖组划分：

| 顶层角色 | 配置值 | 用途 |
| --- | --- | --- |
| 历史任务 | `history` | 第一部分生成成功/失败轨迹、修正候选与来源证据 |
| 目标构建/训练任务 | `target_train` | 第二部分收集 Silent/Remind 干预数据；第三部分训练 |
| 验证集 | `validation` | 内部分 `calibration` 校准收益/风险，以及 `selection` 选策略/阈值 |
| 最终测试集 | `test` | 冻结系统后的未见任务最终评估 |

第一部分的 `collect` 只运行 `history`。同任务的所有复本、同一模板或 `groups` 关联起来的任务必须处于同一分区。`group_id`、`groups` 和 `template_id` 都参与连通分组；已知重复模板不能仅改 ID 后跨集合。`family`/`mechanism` 文字标签不会自动成为分组轴；正式家族/机制留出时，把对应标签显式写入 `groups`。来源留出必须使用不共享来源的独立实验清单与记忆库；固定共享库的探索结果只能说明该库条件下的目标泛化。

支持手工预分配角色，或对未分配任务提供四个角色的 `split_ratios` 自动按独立组分配。比例没有论文默认值，应根据来源记忆产出率、独立任务数量和目标功效估算后确定。`configs/benchmarks/demo.json` 仅展示格式和软件流程，不是足以训练/评估的样本规模。

### 失败采集采用什么停止规则

推荐正式来源采集采用 **固定任务清单 + 每任务固定运行数**，全部保留成功、失败和基础设施错误。这样可以比较任务覆盖率和观察到的失败率，也能利用成功轨迹对照分析失败。不要只保存失败文件而丢失分母。

`CollectionConfig` 支持两种模式：

| 模式 | 停止规则 | 解释 |
| --- | --- | --- |
| `fixed_runs` | 按预先随机化的轮转顺序运行，每任务最多 `runs_per_task`，总计最多 `max_runs` | 推荐主采集；结果不影响停止时机 |
| `failure_target` | 达到 `failure_target` 即停，同时受每任务和总次数上限限制 | 探索性积累数据；结果依赖的停止会改变分布，不能当作无偏失败率样本 |

基础设施错误消耗运行上限，但不增加失败计数；成功轨迹也保留。没有自动补跑或重试。完成的轨迹不可覆盖，相同配置重启会跳过已完成部分；发生中断、且无法确认外部请求是否完成时，会保留 pending 记录并停止自动重试。

**采集预算与来源验证预算分开。** `runs_per_task/max_runs` 决定历史轨迹数量；`replay_pairs=K` 决定每个修正做多少组来源配对回放。当前服务器工程 pilot 采用 K=2；两组必须完整记录，不得因中途收益为正而提前停止。K=2 的结果只能说明这个试跑协议下的修正表现，不能当作统计充分的正式样本量。

### 符号更正

依用户更正，Eq.2 为 `x_t = F_ω(c_t, a0_t, m, e_t, d_t, v_t, ρ_t, b_t)`；小写 `e_t` 仍按 proposal 表示当前原动作对历史机制的暴露。记忆元组 Eq.1 中的大写 `E` 继续表示独立回放/测试证据。没有改写原 PDF 文件，也没有实现或重定义第二部分特征。

## 第一部分的实际流程

```text
任务 catalog → manifest（任务角色、依赖、资产版本）
                    ↓ 只运行 history
               collection（所有历史轨迹）
                    ↓ 成败对照或人工提议
               candidates（教师查看完整公开轨迹并提出检查点/修正）
                    ↓ 冻结候选与原始来源绑定
               固定 K 组来源配对回放
                    ↓ Eq.6 平均效用差
               bank（平均差超过阈值者入库）
```

每个候选必须绑定真实失败轨迹中的动作前检查点及原动作，包含机制假说、修正动作、可执行正/排除谓词、失败签名和来源有效范围。自动提议只接触来源公开观察；成功参照仅来自同一历史任务。手工填写和模型生成的候选使用同一校验与回放流程。自然语言解释本身不会令 `v=1`。

教师候选生成读取失败轨迹的全部公开步骤，可选择其中一个已记录的动作前检查点，并提出机制、替代动作、适用条件和范围；同任务成功轨迹可作为参照。教师只提出假说，不能读取私有测试/隐藏评分材料，也不负责判定修正有效。回放的两个分支从同一检查点分别执行原动作与修正动作，然后使用相同冻结 actor 和续跑预算。所有 K 次计划尝试均保留；发生恢复、调用或无效修正错误时，该候选证据标为未知，不删掉错误后重算一个更好的均值。配对效用差按 Eq.6 计算。

2026-11-15 起新生成的 v6 来源证据：K 组配对全部有效且完成，所有修正续跑均成功，并且 `mean_difference = mean(R(修正) - R(原动作))` 严格大于 `acceptance_epsilon`，才标为 `v=1`。任一配对发生恢复、调用或无效修正错误时，结果为未知。目标配对 pilot 明确要求 K=2。旧 v5 证据继续按其原有平均差规则校验，不改写旧标记，但不能直接进入新目标配对实验。`v=1` 不代表置信区间结论或目标任务上的提醒收益。

同一来源检查点的 K 个复本只支持该检查点和所声明随机协议下的结论，不能冒充 K 个独立来源任务。来源直接替换动作的效用，也不能证明目标任务的文字提醒有效。

## 安装与检查

Python 3.11 或更高。基础包只依赖标准库；真实 Terminal-Bench 执行需要 Linux 和可用 Docker，或显式选择带独立 SquashFS 快照的 Enroot 后端。模型服务可独立部署。

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m cave_mem_v3 check --config configs/phase1/server.toml
.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\Activate.ps1
```

激活环境后可使用后文的 `cave3` 命令；也可始终使用 `.venv\Scripts\python.exe -m cave_mem_v3` 替代。Linux 将 Python 路径换为 `.venv/bin/python`，使用 `source .venv/bin/activate` 激活。测试包含开发环境闭环、注入式模型响应/容器后端及错误路径，不代表已完成真实模型或官方任务实验。

## 本地可检查的开发流程

```text
cave3 manifest --catalog configs/benchmarks/demo.json --output runs/demo/manifest.json
cave3 collect --config configs/phase1/demo.toml --manifest runs/demo/manifest.json --output runs/demo/history
cave3 candidates --collection runs/demo/history --output runs/demo/candidates.json
```

最后一条命令只生成可编辑模板。自动教师模式会查看完整公开轨迹并从中选择检查点；手工模板供逐项审阅。对开发 fixture 中 `history-risk` 的候选，填写：

```json
{
  "mechanism": "Operation before preparation irreversibly fails this fixture.",
  "correction": {"type": "prepare"},
  "positive_conditions": [{"field": "requires_preparation", "equals": true}],
  "exclusion_conditions": [{"field": "prepared", "equals": true}],
  "scope": "Development fixture at the recorded checkpoint only.",
  "proposed_by": "fixture-demo"
}
```

编辑模板已有字段，保留来源 ID、原动作、检查点绑定和自动记录的失败签名。正式任务必须依据实际轨迹提出修正，不能沿用这个 fixture 答案。

```text
cave3 verify --config configs/phase1/demo.toml --manifest runs/demo/manifest.json --collection runs/demo/history --candidates runs/demo/candidates.json --output runs/demo/verification
```

`verify` 会完成每个候选配置好的 K 组配对回放，保留失败或未知证据，并在批次结束后自动将且仅将 `v=1` 候选冻结到 `verification/bank.json`。若需把库写到其他位置，可再用 `cave3 bank --verification runs/demo/verification --output runs/demo/bank.json` 导出；该命令不覆盖已有文件。加载时重新校验来源、修正、回放记录与平均效用差。库文件包含用于重核的 `audit_sources`，不得将整个文件作为 actor 提示词或目标特征输入。

## 服务器真实任务准备

1. 将 `configs/phase1/server.toml` 复制为自己的 `.local.toml`，分别填写冻结 actor 和独立 teacher 的模型服务地址、固定 revision，以及已确定的采集/回放预算。`local_model_path` 是部署记录，实际执行通过 OpenAI-compatible API。API key 只从各自 `api_key_env` 指定的环境变量读取，不写到配置或产物。
2. 固定 Terminal-Bench 2.0 仓库的完整 commit，填写 `configs/benchmarks/terminal.example.json` 中真实任务与分组。清单里的路径可相对于 catalog 文件。支持单容器 Linux 文件任务；持续服务、多容器、网络依赖或不能恢复的任务需要另一个协议。
3. 用 `manifest` 冻结任务资产，先对历史任务执行 `audit`。它会生成未批准的恢复诊断，包括真实文件写入、分支隔离和恢复检查；结合任务依赖审核 `reviewed_constraints`，记录 reviewer 和 review_notes。不能仅把布尔值全部改成 true。
4. 在任务 catalog 的 `protocol` 中设置 `paired_scope="stateless-offline-filesystem-v1"` 与 `audit_path`（相对 catalog 的审核 JSON 路径），再生成新的最终 manifest，然后采集历史轨迹。审计配置要在来源采集前冻结；来源采集后改变协议或 actor，需要新 run。
5. 可用 `candidates --generate --config <server-config>` 调用独立 `[teacher]` 模型自动提议；它不会复用冻结 `[actor]`。再检查候选、固定批次并 `verify` / `bank`。提议模型的每次调用也有费用记录和持久化 journal；解析失败保留费用，未完成请求不自动重发。

```text
cave3 manifest --catalog configs/benchmarks/tasks.local.json --output runs/source/preflight-manifest.json
cave3 audit --config configs/phase1/server.local.toml --manifest runs/source/preflight-manifest.json --task <historical-task-id> --output runs/source/task-audit.json
```

当前 Terminal adapter 是明确的文件系统动作边界协议：每个 shell 动作在新容器运行，只持久化文件和公开历史；`cd`、shell变量、后台进程不跨动作保留。私有官方 tests/solution 不进入 actor 容器，评分在独立容器进行；预算耗尽时也对当前产物评分。它不能直接当作 Harbor 持久 shell 条件下的官方排行榜成绩。恢复审核不证明远端模型随机流、系统时钟或 RNG 完全耦合。

请求的 model/revision、实际返回的 model/fingerprint 都记录在轨迹中。实际模型名必须与请求模型一致，或在 `response_model` 中预先登记服务别名；`expected_fingerprint` 可固定已知部署指纹。服务端不返回指纹时，模型冻结仍依赖已固定的部署与权重清单，代码不会声称获得权重身份认证。所有真实请求只有一次物理尝试，不内置重试。

## 文件职责与运行产物

源码目录 `src/cave_mem_v3/` 按第一部分流程分工：

| 文件 | 内容 |
| --- | --- |
| `common/contracts.py` | 公开上下文、来源记忆 `m`、效用、版本哈希与 JSON 读写 |
| `common/config.py` | 配置加载、未填写服务器参数检查 |
| `benchmarks/tasks.py` | 四角色 manifest、依赖组划分、资产冻结与校验 |
| `memory_generation/collection.py` | 历史任务调度、两种停止规则、全量保存及续跑 |
| `agents/providers.py` | 冻结 actor、OpenAI-compatible 请求、解析与费用记录 |
| `environments/base.py` | 环境接口和明确标注的开发 fixture |
| `environments/terminal.py` | Terminal-Bench 文件系统任务、检查点恢复及隔离评分 |
| `environments/enroot.py` | H20 节点的 Enroot 镜像快照后端；并非 Harbor 会话等价实现 |
| `agents/runtime.py` | 单条历史执行/来源回放续跑，统一计量与异常分类 |
| `memory_generation/candidates.py` | 真实失败绑定、候选模板和模型结构化提议 |
| `memory_generation/predicates.py` | 公开条件的可执行表达与三值判断 |
| `memory_generation/replay.py` | 冻结 K 组配对回放计划、Eq.6 平均效用差与证据重算 |
| `memory_generation/bank.py` | 已验证记忆构建、主库冻结和加载复核 |
| `cli.py`、`__main__.py`、`__init__.py` | 命令入口和包版本 |

`configs/` 有开发配置、未完成的服务器配置和对应任务模板；`tests/` 检验关键研究合同和可执行链路。[实验设计](experiment_design.md) 保留四阶段规划；[变更记录](../CHANGELOG.md) 逐文件记录迁移和修改。

历史 run 包含 `run.json`（冻结计划）、`traces/`（每次成功/失败/错误）、`summary.json`（次数、覆盖与停止原因）。候选模型另有 `<candidates文件>.generation/` 调用记录。验证 run 保存 `source_traces.json`、`candidates.json`、逐候选 `plan.json`/`pairs/`/`evidence.json`、汇总 `verification.json` 及通过时生成的 `bank.json`。库中保存记忆、证据质量和审计来源。运行产物、环境、权重与密钥不进入版本控制。

H20 历史工程 pilot 的任务集合、镜像和结果保存在忽略的 `data/`、`runs/`。此前根据单容器文件任务和不同依赖组，有目的地预选并冻结 3 道 history（fix-git、regex-log、sqlite-db-truncate），不是从 89 道题随机抽取；其余 target_train、validation calibration/selection、test 各按任务依赖组隔离且不运行。旧 r3 采集每题一次、最多 6 步/6 次 actor 调用；新的 20 步配置必须开始独立 run，不能套用旧检查点。教师对每条有效失败轨迹最多提 1 个候选。Enroot 将 `/tmp` 在动作边界同步回容器文件系统，官方 verifier 在独立容器运行并使用临时 tests 副本。恢复诊断可通过，但官方 `test.sh` 运行时联网安装依赖，任务范围审计不标为正式批准。

`r2` 的格式错误（不合法 JSON）没有执行 shell，也没有官方任务分数，不与官方测试失败混为一谈。独立教师 3 次请求均完成，3 条自动候选经 K=2 回放全部未通过，暂定库 0。为减少 actor 输出协议导致的非任务失败，`r3` 从新的历史 run 开始为本地 Qwen 服务显式请求 JSON 对象模式；自动教师生成只使用已执行且有明确任务结局的失败，另记排除的来源 ID。教师配置和官方评分均不改变。镜像内缺少命令属于可见的任务环境事实，actor 重复使用该命令仍是策略失败；Enroot 钩子产生的警告另作为适配器局限记录，不能解释为任务测试失败。

`r3` 的 3 条历史轨迹均由官方测试判失败，0 基础设施异常、0 JSON 解析错误；独立教师生成 3 条候选，K=2 来源回放共 6 组，`regex-log` 两组修正成功且效用差各 +1，另两候选未通过。历史文件 `runs/pilot-20261114-r3/pilot-provisional-bank.json` 是旧 pilot 脚本输出的临时工程产物，不是 proposal 要求的第二种库，也不是正式验证结果；正式 `verify`/`bank` 流程只把 `v=1` 候选冻结进主库。没有运行预留的 target/validation/test 任务，也不据此主张泛化或统计置信结论。详细计数及 token 成本见 [变更记录](../CHANGELOG.md)。

旧 pilot 脚本曾把 `pilot-provisional-bank.v1` 写成独立工程产物：K=2 两组都修正成功且效用差严格大于零才保留该项。旧结果不按新规则重新标注。当前 `verify` 固定执行配置指定 K，完成后按平均效用差赋 `v`，通过项自动进入 `bank.json`。K=2 不构成置信结论或跨任务泛化证据。教师网关只暴露模型别名，配置中的 alias/revision 记录不等于权重证明；费用若无价格配置仅报告 token 数，不推算货币成本。

本次 H20 一次性作业使用 `scripts/phase1_actor_service.py <run-name> collect` 和 `scripts/phase1_actor_service.py <run-name> pilot`；后者调用 `scripts/phase1_pilot_replay.py`。脚本需要 Slurm H20 作业、预先冻结的 `runs/<run-name>/manifest-final.json`、本地权重与镜像；模型服务结束时关闭进程组。教师生成仍用独立的 `cave3 candidates --generate`，只在有密钥的 head 环境调用。
