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

第二部分预计在包内增加 `interventions/`，第三部分增加 `policy/`，第四部分增加 `evaluation/`；对应配置按阶段加入 `configs/`，文档加入 `docs/`。这些是规划路径，当前没有相关占位实现。

公共模型与环境代码继续放在 `agents/` 和 `environments/`；第一部分的 `memory_generation/` 只负责来源记忆。目标提醒的结果、训练标签与测试结果在后续各阶段产生，不能倒流为来源采集或原动作的输入。
