# CAVE-Mem-v3

**Learning When to Intervene with Failure Memories**

动作条件化、风险约束的失败记忆干预研究。冻结任务执行模型，通过历史失败证据与目标任务随机实验，学习何时提供提醒以及何时保持沉默。

当前已实现第一部分，以及第二部分的可恢复配对和在线随机单臂采集：**历史采集 → 来源验证 → 冻结记忆库 → 缓存目标动作与状态 → Silent/Remind → 记录结局与成本**。配对路线记录两臂结果；单臂路线在同一个运行中的环境里只执行随机分配的一臂。第二部分尚无真实目标提醒实验结果。

[项目架构](docs/architecture.md) · [第一部分使用指南](docs/phase1.md) · [服务器交接](docs/server_handoff.md) · [完整实验设计](docs/experiment_design.md) · [逐文件变更](CHANGELOG.md)

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
| 3. 训练与选择 | 估计收益/风险，比较监督与约束策略 | 已设计，待实现 |
| 4. 部署与评估 | 冻结策略，评估未见任务的成功、回退与成本 | 已设计，待实现 |

`x_t` 中使用用户更正后的小写 `e_t`；记忆元组中的大写 `E` 保持来源证据含义。任务执行模型可采用 `Qwen/Qwen2.5-7B-Instruct`，当前服务器配置保留空的 endpoint、revision、权重路径和未确定的运行次数。

## Quick Start

需要 Python 3.11+。以下入口可从仓库直接运行，无需模型权重或 API：

```bash
python scripts/run_tests.py
python examples/phase1_demo.py --output runs/phase1-demo
python examples/phase2_demo.py --output runs/phase2-demo
python examples/phase2_single_demo.py --output runs/phase2-single-demo
```

示例仅使用开发 fixture，产出可检查的轨迹、候选、回放证据和 `bank.json`。相同输出目录不能覆盖重跑；更换 `--output` 即可保留每次结果。

第二部分的配对示例额外生成 `prepared/` 和 `paired/`，单臂示例生成 `single/`。已有 `prepare-target` 产物可使用同一冻结配置运行：

```bash
cave3 pair-target --config <frozen.toml> --target-run <prepared-directory> --bank <bank.json> --output <new-paired-directory>
cave3 summarize-target-pairs --run <paired-directory>
```

仅接收 `target_train`、同一完整 manifest 的历史来源库及新规则 K=2 证据。两臂均执行，随机化的是执行顺序；Silent 执行缓存动作，Remind 只提示一次。两臂保留相同的普通续跑预算，Remind 额外使用预留的一次调用。中断或请求完成情况不明时保留记录并停止，重复命令不会自动重试。摘要从原始事件重算，成本未知时保留结局但不生成效用差。

不可配对时，使用在线单臂入口从任务起点运行；先冻结 `a0/x_t`，然后随机分配一次并继续原环境。它不读取 `prepare-target` 目录，也不调用目标快照或恢复接口：

```bash
cave3 collect-single-target --config <frozen.toml> --manifest <manifest.json> --bank <bank.json> --output <new-single-directory> --role target_train --encoder-path <local-encoder-directory> --encoder-revision <pinned-revision> --assignment-probability 0.5
cave3 summarize-target-single --run <single-directory>
```

`--assignment-probability` 是预先固定的 `P(Remind)`，必须显式填写且严格介于 0 和 1；检查点纳入概率另存。`--exposure-rules` 可指定已有暴露规则，缺失时保留 `e=?`。单臂仅保存观察到的结局和效用及分配概率，不生成配对 D/H/rescue。无记忆或预算不足的检查点不随机化，保留费用和排除原因，未知终局不算失败。完成后的续读和只读汇总均不发出模型请求；未完成会话无法自动恢复或重试。现有真实 Terminal 入口仍受文件系统协议及任务范围审核约束，新增真实后端需独立适配。

安装可用的 `cave3` 命令：

```bash
python -m pip install -e .
cave3 check --config configs/phase1/server.toml
```

服务器模板返回 `ready: false` 是预期结果，需先填写部署与预算参数。实际任务的分步操作、benchmark 划分、采集停止规则、恢复审核和证据范围见[第一部分指南](docs/phase1.md)。运行产物存放于被 Git 忽略的 `runs/`。
