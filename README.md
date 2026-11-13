# CAVE-Mem-v3

**Learning When to Intervene with Failure Memories**

动作条件化、风险约束的失败记忆干预研究。冻结任务执行模型，通过历史失败证据与目标任务随机实验，学习何时提供提醒以及何时保持沉默。

当前已实现第一部分的完整软件流程：**历史采集 → 修正候选 → 独立回放验证 → 冻结记忆库**。真实模型、容器及正式 benchmark 实验仍待服务器运行。

[项目架构](docs/architecture.md) · [第一部分使用指南](docs/phase1.md) · [完整实验设计](docs/experiment_design.md) · [逐文件变更](CHANGELOG.md)

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
| 1. 历史失败记忆 | 采集来源轨迹，验证修正，冻结记忆库 | 已实现；77 项离线测试通过 |
| 2. 提醒实验数据 | 缓存原动作，随机化 Silent/Remind，记录目标结果 | 已设计，待实现 |
| 3. 训练与选择 | 估计收益/风险，比较监督与约束策略 | 已设计，待实现 |
| 4. 部署与评估 | 冻结策略，评估未见任务的成功、回退与成本 | 已设计，待实现 |

`x_t` 中使用用户更正后的小写 `e_t`；记忆元组中的大写 `E` 保持来源证据含义。任务执行模型可采用 `Qwen/Qwen2.5-7B-Instruct`，当前服务器配置保留空的 endpoint、revision、权重路径和未确定的运行次数。

## Quick Start

需要 Python 3.11+。下列两个入口可从仓库直接运行，无需模型权重或 API：

```bash
python scripts/run_tests.py
python examples/phase1_demo.py --output runs/phase1-demo
```

示例仅使用开发 fixture，产出可检查的轨迹、候选、回放证据和 `bank.json`。相同输出目录不能覆盖重跑；更换 `--output` 即可保留每次结果。

安装可用的 `cave3` 命令：

```bash
python -m pip install -e .
cave3 check --config configs/phase1/server.toml
```

服务器模板返回 `ready: false` 是预期结果，需先填写部署与预算参数。实际任务的分步操作、benchmark 划分、采集停止规则、恢复审核和证据范围见[第一部分指南](docs/phase1.md)。运行产物存放于被 Git 忽略的 `runs/`。
