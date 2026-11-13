# 变更记录

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
