# 第一部分：历史失败记忆构建

[返回项目首页](../README.md) · [项目架构](architecture.md) · [完整实验设计](experiment_design.md)

本文命令均从仓库根目录运行。配置已按 `configs/phase1/` 与 `configs/benchmarks/` 分类。

本项目当前实现 proposal §3.1 的第一部分：历史任务采集、失败修正假说、独立来源回放和冻结记忆库。第二部分的目标提醒实验、第三部分的训练、第四部分的部署尚未实现。

执行模型可以选择 **Qwen/Qwen2.5-7B-Instruct**。它承担冻结 actor 的角色，训练对象在后续阶段才是提醒控制器。当前不下载模型；服务器配置的 endpoint、revision、模型本地路径和未确定的样本量保持空缺。读取配置或跑软件测试不会请求模型。

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

**采集预算与来源验证预算分开。** `runs_per_task/max_runs` 决定历史轨迹数量；`screening_pairs/confirmation_pairs` 是每个修正的来源回放对数。后者不能“通过了就提前停”。先用独立探索 pilot 估计方差、最小有意义改善及计算费用，再填写正式预算。本次服务器模板刻意不替你决定这些数字。

### 符号更正

依用户更正，Eq.2 为 `x_t = F_ω(c_t, a0_t, m, e_t, d_t, v_t, ρ_t, b_t)`；小写 `e_t` 仍按 proposal 表示当前原动作对历史机制的暴露。记忆元组 Eq.1 中的大写 `E` 继续表示独立回放/测试证据。没有改写原 PDF 文件，也没有实现或重定义第二部分特征。

## 第一部分的实际流程

```text
任务 catalog → manifest（任务角色、依赖、资产版本）
                    ↓ 只运行 history
               collection（所有历史轨迹）
                    ↓ 成败对照或人工提议
               candidates（可证伪修正假说）
                    ↓ 冻结候选与原始来源绑定
               screening → independent confirmation
                    ↓ Eq.6 效用差与置信区间
               verified bank（仅 v=1）
```

每个候选必须绑定真实失败轨迹中的动作前检查点及原动作，包含机制假说、修正动作、可执行正/排除谓词、失败签名和来源有效范围。自动提议只接触来源公开观察；成功参照仅来自同一历史任务。手工填写和模型生成的候选使用同一校验与回放流程。自然语言解释本身不会令 `v=1`。

回放的两个分支从同一检查点分别执行原动作与修正动作，然后使用相同冻结 actor 和续跑预算。筛选和确认使用不同种子及固定计划；所有确认尝试保留，有恢复/调用错误时标为未知，不删除错误后计算一个更好的均值。确认效用为 `Y - λC*C - λT*T`；默认示例以模型调用数计 C、工具调用数计 T，同时记录绝对 token、费用和耗时，未知费用保持 null。

当前证据区间使用有界效用的 Hoeffding 界，并按本批候选数调整置信水平。若每臂效用范围宽度为 W、候选数 M、置信水平 1−α、确认对数 K，半径为 `W*sqrt(2*log(2*M/α)/K)`。仅下界大于预设 `acceptance_epsilon` 时接受。这个界可能保守，16 或 32 对不意味着一般任务已有足够证据；开发 fixture 中的强修复效应不能据此外推。

同一来源检查点的 K 个复本只支持该检查点和所声明随机协议下的结论，不能冒充 K 个独立来源任务。来源直接替换动作的效用，也不能证明目标任务的文字提醒有效。

## 安装与检查

Python 3.11 或更高。基础包只依赖标准库；真实 Terminal-Bench 执行另需 Linux 和可用 Docker，模型服务可独立部署。

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

最后一条命令只生成可编辑模板。对开发 fixture 中 `history-risk` 的候选，填写：

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
cave3 bank --verification runs/demo/verification --output runs/demo/bank.json
```

失败或未知证据保留在 verification 中；仅接受的候选进入主库。`bank` 不覆盖已有文件；加载时重新校验来源、修正、回放记录、效用与区间。库文件包含用于重核的 `audit_sources`，不得将整个文件作为 actor 提示词或目标特征输入。

## 服务器真实任务准备

1. 将 `configs/phase1/server.toml` 复制为自己的 `.local.toml`，填写 actor 的模型服务地址、固定 revision 和已确定的采集/回放预算。`local_model_path` 是部署记录，实际执行通过 OpenAI-compatible API。API key 只从 `api_key_env` 指定的环境变量读取，不写到配置或产物。
2. 固定 Terminal-Bench 2.0 仓库的完整 commit，填写 `configs/benchmarks/terminal.example.json` 中真实任务与分组。清单里的路径可相对于 catalog 文件。支持单容器 Linux 文件任务；持续服务、多容器、网络依赖或不能恢复的任务需要另一个协议。
3. 用 `manifest` 冻结任务资产，先对历史任务执行 `audit`。它会生成未批准的恢复诊断，包括真实文件写入、分支隔离和恢复检查；结合任务依赖审核 `reviewed_constraints`，记录 reviewer 和 review_notes。不能仅把布尔值全部改成 true。
4. 在任务 catalog 的 `protocol` 中设置 `paired_scope="stateless-offline-filesystem-v1"` 与 `audit_path`（相对 catalog 的审核 JSON 路径），再生成新的最终 manifest，然后采集历史轨迹。审计配置要在来源采集前冻结；来源采集后改变协议或 actor，需要新 run。
5. 可用 `candidates --generate --config <server-config>` 调用配置的模型自动提议，再检查候选、固定批次并 `verify` / `bank`。提议模型的每次调用也有费用记录和持久化 journal；解析失败保留费用，未完成请求不自动重发。

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
| `environments/terminal.py` | Docker 文件系统任务、检查点恢复及隔离评分 |
| `agents/runtime.py` | 单条历史执行/来源回放续跑，统一计量与异常分类 |
| `memory_generation/candidates.py` | 真实失败绑定、候选模板和模型结构化提议 |
| `memory_generation/predicates.py` | 公开条件的可执行表达与三值判断 |
| `memory_generation/replay.py` | 冻结回放计划、独立确认、Eq.6 证据重算 |
| `memory_generation/bank.py` | 已验证记忆构建、主库冻结和加载复核 |
| `cli.py`、`__main__.py`、`__init__.py` | 命令入口和包版本 |

`configs/` 有开发配置、未完成的服务器配置和对应任务模板；`tests/` 检验关键研究合同和可执行链路。[实验设计](experiment_design.md) 保留四阶段规划；[变更记录](../CHANGELOG.md) 逐文件记录迁移和修改。

历史 run 包含 `run.json`（冻结计划）、`traces/`（每次成功/失败/错误）、`summary.json`（次数、覆盖与停止原因）。候选模型另有 `<candidates文件>.generation/` 调用记录。验证 run 保存 `source_traces.json`、`candidates.json`、逐候选 `plan.json`/两阶段回放/`evidence.json` 和汇总 `verification.json`。最终 `bank.json` 保存记忆、证据质量和审计来源。运行产物、环境、权重与密钥不进入版本控制。
