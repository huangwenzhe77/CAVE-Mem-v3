# CAVE-Mem-v3

**Learning When to Intervene with Failure Memories**

动作条件化、风险约束的失败记忆干预研究。冻结任务执行模型，通过历史失败证据与目标任务随机实验，学习何时提供提醒以及何时保持沉默。

当前已实现四部分核心软件：**历史采集 → 来源验证与冻结记忆库 → 目标配对/随机单臂 → 训练与独立校准 → 约束策略 → 完整验证选型 → 最终冻结 test 与报告**。整体验收状态见任务卡。所有本轮演示和测试均为离线 fixture，尚未产生本轮真实目标提醒实验结果。

2026-11-16 软件验收：353 项测试、两项独立审核及实际拟合学习策略的完整离线示例通过。本轮改动在本地 `codex/phase3-goal-workflow`，尚未提交/推送；远端旧 main/s2 分支不包含这些新增实现。

[项目架构](docs/architecture.md) · [Proposal/流程图代码追踪表](docs/traceability.md) · [第一部分](docs/phase1.md) · [训练](docs/policy.md) · [冻结评估](docs/evaluation.md) · [新软件交接](docs/software_handoff.md) · [实验设计](docs/experiment_design.md) · [逐卡记录](docs/task_cards.md) · [逐文件变更](CHANGELOG.md)

## Project Structure

```text
CAVE-Mem-v3/
├── configs/                         # 实验配置与 benchmark 任务清单
│   ├── benchmarks/                  # 任务角色、依赖组与资产路径
│   └── phase1/                      # 历史记忆构建配置
├── docs/                            # 架构、实验设计与操作指南
├── examples/                        # 可直接运行的完整示例
│   └── phase1_demo.py               # 无联网的第一部分开发演示
├── scripts/                         # 开发辅助入口
│   └── run_tests.py                 # 跨平台离线测试
├── src/cave_mem_v3/                  # 可安装的研究代码包
│   ├── agents/                      # 冻结 actor、模型接口与执行循环
│   ├── benchmarks/                  # 任务清单、分组与资产校验
│   ├── common/                      # 数据合同、效用、配置与持久化
│   ├── environments/                # 环境接口、fixture 与 Docker 后端
│   ├── memory_generation/           # 第一部分：采集、候选、验证、记忆库
│   ├── target_data/                 # 第二部分：检索、采样、冻结状态与角色
│   ├── interventions/               # 第二部分：配对/单臂执行与原始记录审核
│   ├── policy/                      # 第三四部分：训练、校准、选型与最终评估
│   ├── cli.py                       # cave3 命令
│   └── __main__.py                  # python -m cave_mem_v3
├── tests/                           # 实验合同与完整流程测试
├── CHANGELOG.md                     # 文件变更与验证记录
├── pyproject.toml                   # 包定义与依赖
└── README.md
```

目录参考 [SkillRL](https://github.com/huangwenzhe77/SkillRL) 的职责分组方式，并采用 `src` 包布局。`memory_generation/` 对应第一部分；公共执行与环境模块供后续阶段复用。详细文件映射与选择理由见[架构说明](docs/architecture.md)。

## Research Roadmap

| 部分 | 目标 | 当前状态 |
| --- | --- | --- |
| 1. 历史失败记忆 | 采集来源轨迹，验证修正，冻结记忆库 | 软件已实现；新证据要求全部修正成功且平均效用差超阈值 |
| 2. 提醒实验数据 | 缓存原动作，采集配对或随机单臂结果 | 两条采集路线已实现；真实实验与自然/四格覆盖待完成 |
| 3. 训练与选择 | 估计收益/风险，比较监督与约束策略 | 数据入口、建模/校准、监督 gate、DR、约束 bandit、配对/保守选型已实现 |
| 4. 部署与评估 | 冻结策略，评估未见任务的成功与成本 | 自包含冻结包、完整 validation/test、组分配与统计已实现；test 个体回退仍需独立配对实验 |

`x_t` 中使用用户更正后的小写 `e_t`；记忆元组中的大写 `E` 保持来源证据含义。任务执行模型可采用 `Qwen/Qwen2.5-7B-Instruct`。仓库的服务器配置保留过往工程 pilot 设置，正式部署仍需核实实际服务、权重版本、预算和任务资产，不能直接当作已审核的实验配置。

## Quick Start

需要 Python 3.11+。先安装本地包和训练模块的数值依赖。以下入口无需模型权重或 API：

```bash
python -m pip install -e '.[policy]'
python scripts/run_tests.py
python examples/phase1_demo.py --output runs/phase1-demo
python examples/phase2_demo.py --output runs/phase2-demo
python examples/phase2_single_demo.py --output runs/phase2-single-demo
python examples/four_phase_learning_demo.py --output runs/four-phase-learning-demo
```

示例仅使用开发 fixture，产出可检查的轨迹、候选、回放证据和 `bank.json`。相同输出目录不能覆盖重跑；更换 `--output` 即可保留每次结果。

真实教师输出若截断或解析失败，可在提高冻结的教师输出预算后使用新输出路径执行 `cave3 candidates --generate --collection <history> --config <new-teacher.toml> --reuse-successes-from <old-candidates.json> --output <new-candidates.json>`。它校验并复用同一历史批次已成功的每条唯一候选，只对失败项发起一次新请求；旧费用和失败记录保留，不产生同一历史轨迹的多个有效候选。

真实 Terminal-Bench 的官方测试可能在容器内下载依赖，超过任务清单中的 600 秒执行保护。Enroot 运行时可设置 `CAVE_VERIFIER_TIMEOUT_FLOOR_SECONDS=1800`，仅延长宿主机等待私有测试结束的时间；任务、测试脚本、奖励判定、actor 配置及冻结任务协议保持原样，实际等待上限写入私有验证结果。重跑仍需使用新的输出目录以保留原始超时证据。

`four_phase_learning_demo.py` 进一步生成实际拟合的收益/风险模型、校准、监督 gate、配对与单臂 DR bandit，冻结候选后运行新完整验证，最后比较选中学习策略与真正不检索记忆的 actor 基线。输出包含原始记录、选择锁、最终计划/报告和 `summary.json`。脚本模型、两个公开玩具条件及宽松演示阈值仅用于验收；未实现真实模型效果证明或正式研究参数选择。单臂 bandit 仍借助兼容的配对初始化与校准，不代表纯单臂数据可从零完成整套训练。

第二部分的配对示例额外生成 `prepared/` 和 `paired/`，单臂示例生成 `single/`。已有 `prepare-target` 产物可使用同一冻结配置运行：

```bash
cave3 pair-target --config <frozen.toml> --target-run <prepared-directory> --bank <bank.json> --output <new-paired-directory>
cave3 summarize-target-pairs --run <paired-directory>
```

接收 `target_train` 或明确的 validation 子角色、同一完整 manifest 的历史来源库及新规则 K=2 证据。两臂均执行，随机化的是执行顺序；Silent 执行缓存动作，Remind 只提示一次。两臂保留相同的普通续跑预算，Remind 额外使用预留的一次调用。中断或请求完成情况不明时保留记录并停止，重复命令不会自动重试。摘要从原始事件重算，成本未知时保留结局但不生成效用差。

不可配对时，使用在线单臂入口从任务起点运行；先冻结 `a0/x_t`，然后随机分配一次并继续原环境。它不读取 `prepare-target` 目录，也不调用目标快照或恢复接口：

```bash
cave3 collect-single-target --config <frozen.toml> --manifest <manifest.json> --bank <bank.json> --output <new-single-directory> --role target_train --encoder-path <local-encoder-directory> --encoder-revision <pinned-revision> --assignment-probability 0.5
cave3 summarize-target-single --run <single-directory>
```

若同一任务清单中既有可恢复又有不可恢复检查点，使用 `collect-auto-target`。它保留选中的活会话，先对该点做独立恢复审核：通过则从同一检查点执行 Silent/Remind 配对；未通过则在未执行原动作的活会话随机分配并执行一臂。分流决定及失败审核均保存；任一分支开始后不再改换路线，未确认完成的外部请求不自动重试。单臂概率必须预先固定：

```bash
cave3 collect-auto-target --config <frozen.toml> --manifest <manifest.json> --bank <bank.json> --output <new-auto-directory> --role target_train --encoder-path <local-encoder-directory> --encoder-revision <pinned-revision> --assignment-probability 0.5
cave3 summarize-target-auto --run <auto-directory>
cave3 judge-paired-exposure --run <auto-directory> --config <frozen.toml> --output <new-exposure-directory>
cave3 build-policy-dataset --auto <auto-directory> --exposure <auto-directory> <exposure-directory> --output <new-dataset.json>
```

`--assignment-probability` 是预先固定的 `P(Remind)`，必须显式填写且严格介于 0 和 1；检查点纳入概率另存。配对执行后，独立教师只看历史机制、目标检查点和 Silent 原动作的公开执行结果，给出事后 `e=0/1/?` 与理由；数据集把它保存在 `e_observed` 和版本化 `x_final`，不把 Remind 结局或任务评分交给教师。单臂不调用此教师，也没有单独的事后 `e` 标签，记录 `x,z,r,b`。原冻结 `x` 仍用于在线决策；`x_final` 是采集后的结果字段。教师输出不合法或证据不足会保留错误或 `?`，不会编造标签。`--exposure-rules` 指定原有动作前估计规则，缺失时原冻结 `x` 的旧估计值仍为 `?`。无记忆或预算不足的检查点不随机化，保留费用和排除原因，未知终局不算失败。完成后的续读和只读汇总均不发出模型请求；未完成会话无法自动恢复或重试。现有真实 Terminal 入口仍受文件系统协议及任务范围审核约束，新增真实后端需独立适配。

若冻结来源 manifest 的目标任务尚无审核，可在目标运行配置的 `[environment]` 设置 `audit_map_path = "/absolute/path/to/audit-map.json"`。该 JSON 将任务 ID 映射到各自经恢复诊断和范围审核的 audit 记录；记录内容进入目标运行的环境配置，原来源 manifest 与库哈希不改动。`cave3 audit --task <target-id>` 可生成待审核诊断，需填写真实 reviewer、review_notes 并核实任务范围后才能用于映射；重复或不匹配的审核在采集前拒绝。

第三部分包括只读数据入口、配对收益/风险/成本模型、组级校准、监督 gate、单臂 DR 与约束 bandit，使用见 [训练指南](docs/policy.md)。选型和最终执行见 [冻结评估](docs/evaluation.md)。配对输入需要同时保留原始 prepared 目录；加载时重新核对原始记录和任务资产。下列两个参数可重复填写，但只允许合并相同 manifest、来源库、actor、encoder 和效用设置的运行：

```bash
cave3 build-policy-dataset --paired <paired-directory> <prepared-directory> --single <single-directory> --output <new-dataset.json>
```

只提供其中一种路线也可导出；默认入口只接收 target_train。特征、标签、概率、成本和分组来源分开记录，单臂不会生成配对伤害标签；未知费用、错误和排除记录保留。后续软件目标与逐卡验收见 [软件落地目标](docs/software_goal.md) 和 [任务卡](docs/task_cards.md)。

独立验证任务沿用完整 manifest 与冻结历史库。`prepare-target` 或 `collect-single-target` 需显式指定 `--role validation --validation-role calibration`，选型任务则指定 `selection`；随后 `pair-target` 从 prepared 继承角色。导出数据时同样显式指定角色：

```bash
cave3 build-policy-dataset --paired <calibration-paired> <calibration-prepared> --role validation --validation-role calibration --output <new-calibration-dataset.json>
```

每份数据只属于训练、校准、选型中的一个分区；默认训练加载器拒绝验证数据。旧训练产物仍按 v1 重核，含糊的旧 validation 记录不能改标使用。以上命令尚未进行真实采集。

安装可用的 `cave3` 命令：

```bash
python -m pip install -e .
cave3 check --config configs/phase1/server.toml
```

`check` 只检查配置必填值，本机现有配置返回 `ready: true`、`model_downloaded: false`；它不证明模型/API、任务资产、容器或实验参数已经准备完成。实际任务的分步操作、benchmark 划分、采集停止规则、恢复审核和证据范围见[第一部分指南](docs/phase1.md)。运行产物存放于被 Git 忽略的 `runs/`。
