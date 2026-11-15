# 两个 agent 的方案讨论记录

日期：2026-11-15。性质：只读研究/工程审阅及交叉讨论，不是实验结果。

[方案](v1_plan.md) · [任务卡](task_cards.md)

## 参与与过程

- `research_review`：核查提案的研究目标、来源证据、策略/采集概率、配对识别和阶段边界。
- `engineering_review`：核查工作区的状态生成、runtime、恢复、actor 接口、预算与测试缺口。
- 两个 agent 先独立审阅，再相互发送意见，讨论 `π/b/order`、恢复与单臂边界、K=2 入库规则以及预算。主 agent 汇总并对照源码与 PDF。
- 两个 agent 均未编辑文件、调用真实模型或运行 benchmark。本轮代码修改数为零。

基线是 commit `ee97ac12d4e43d4f168e6ce2acb5f2196c1a56f5` 加已有未提交工作区；不能把未提交能力归入该 commit。

## 讨论与取舍

| 议题 | 研究审阅意见 | 工程审阅意见 | 本版结论 |
| --- | --- | --- | --- |
| V1 的截止点 | 先建立可信干预数据，不预设 RL 优势 | 恢复、提醒、日志尚未完成 | 以可恢复单检查点配对闭环为 V1；训练/最终测试延期 |
| `x_t` 后是否先跑 gate | `πθ` 属于未来学习策略，不等于采集概率 | 当前没有 gate 权重或输出合同 | 先定义输出合同；V1 采集不伪造收益/风险预测 |
| 配对随机化 | 两臂都观察时需明确顺序与耦合 | 实现必须区分 pair order 和单臂 assignment | 冻结随机顺序，单列 `order_probability`；不写成 `b(z|x)` |
| 目标资格 | 筛掉 `(ρ=1,e=0)` 会删掉核心研究问题 | 资格检查应只核验来源、技术和预算 | 不按 `ρ=e=1` 筛主样本，未知也保留 |
| 恢复失败 | 不能据不可靠 forks 声称配对回退 | prepare 后环境会关闭，单臂不是换个标志即可支持 | 保存阻塞；不可恢复活会话 adapter 延期，不自动降级 |
| 来源门槛 | 交接字面与代码均值规则不等价 | v5 evidence 和 bank 重核需要版本兼容 | 用户本轮决定新 K=2：两修正均成功且平均差超 epsilon；旧结果保留 |
| 调用预算 | 必须计入草稿与额外提醒调用，续跑额度可比 | 若总上限不变，应预留重提调用槽 | 推荐两臂统一 `R−1` 普通续跑额度，Remind 用额外预留槽；V1-01 固化字段 |
| 真实环境 | 旧 pilot 不等于正式协议已认证 | verifier 联网依赖超出当前离线协议 | 先审核固定测试依赖镜像方案，再做真实恢复审核 |

两位 agent 在 V1 范围与研究边界上没有保留相反建议。模型/任务具体版本、pilot 样本量、成本权重和服务器部署仍属于运行前必须填齐的参数，不能由讨论记录假装已经确定。

## 可定位的核查发现

以下行号对应本轮审阅时工作区，后续实现可能移动。

1. `agents/runtime.py:124`：选中节点后先保留动作并构建状态，`pre_treatment_ready` 在动作验证/执行前退出。`target_data/pre_treatment_state.py:394` 明确 `not_assigned`，下一行是 `outcome_access=none`。
2. `agents/runtime.py:132`：剩余环境步数写为 `min(spec.max_steps, context.remaining_steps)-index`。`environments/terminal.py:402` 的公开剩余步数已经扣除执行步数，因此晚节点会重复扣减。
3. 工程 agent 用只读、无文件写入的 fixture 探针复现：`max_steps=4`、选中 `index=1`、公开剩余 `3`，输出预算却为 `2`。这是软件观察。建议修复表达式为 `min(spec.max_steps-index, context.remaining_steps)`，还须测试研究与环境上限不一致的情形。
4. `memory_generation/replay.py:288` 与 `:295`：当前 v5 规则仅用完整有效配对及平均效用差，不额外要求每组修正成功。新规则不能直接改写旧证据的重算语义。
5. `target_data/collection.py:172`：准备结束关闭 actor/env。`environments/terminal.py:546` 保留内容寻址镜像，但仍需验证真实资产可恢复，不能只相信 JSON 中有 checkpoint 字段。
6. `environments/terminal.py:358`：认证依赖审计、镜像、资产和文件隔离；不等于任意 Terminal-Bench 任务或远端 RNG 恢复保证。
7. `agents/providers.py:263` 附近：来源 actor 提示渲染器拒绝 reminder。需要目标专用入口，不能通过解除来源保护实现提醒。
8. `docs/phase1.md` 的 H20 段落记录 verifier 运行时联网安装依赖、范围审核未正式批准。真实新 pilot 必须处理该不兼容。

## 本轮验证和证据范围

两个 agent 对初稿再次复核后，已补正：提醒渲染器先于计划冻结；pair/arm 持久日志由编排器负责，CLI 只调度；镜像/adapter 缺失须另卡完成；任务级恢复审核不替代实际目标检查点逐点审核；两臂对 actor 无效动作采用相同处理口径。

主 agent 执行 `python scripts/run_tests.py`：139 项通过。PDF 全文提取，并渲染核对第 2、3 页的公式与干预定义。提案的 Eq.2 在 PDF 里仍为大写 `E`，实现计划依用户确认更正为小写 `e_t`；没有编辑提案。

本轮没有新真实 benchmark 观察；历史 H20 结果只作为已有文档中的背景。测试通过不能证明恢复在服务器成立、提醒有效或策略已训练。

## 形成的交付

- `v1_plan.md`：V1 范围、接口/研究规则、里程碑及延后范围。
- `task_cards.md`：21 张 V1 小任务卡和 6 张后续设计卡，每张包含 Goal、Context、Rules、Acceptance。
- 根目录 `AGENTS.md` 的执行约定：后续一轮只执行一张明确卡，不自动推进下一卡。
