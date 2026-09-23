# 待办清单

按优先级排列。每一条都写清了「为什么」和「怎么验证」——一个没有验证方式的待办，无法判断它是否真的做完了。

图例：**P0** 阻塞交付 · **P1** 影响结论可信度 · **P2** 体验与扩展

---

## P0 — 出数据之前必须做

### 1. 跑完整消融矩阵（4 配置 × 5 任务 = 20 cell）

```bash
.venv/bin/python -m benchmark.driver --configurations vanilla m mr mrsx
```

- **为什么还没跑**：约 200 万 token 的配额消耗，这个决定应由你来做，我没有替你花。
- **单 cell 参考**：`lj_melt × mrsx` 实测 310 秒、39 次工具调用、约 10.5 万 token。
- **预计**：1.5–2 小时，顺序执行（共用配额与超算账号，并发会让归因失效）。
- **产出**：`benchmark/runs/<cell>/result.json` 与 `benchmark/runs/summary.json`。
- **验证**：`summary.json` 里四个配置的 `compliance_rate` 应呈现单调性；若 `vanilla` 与 `mrsx` 无差异，说明 adapter 没起作用，需要先查 overlay 是否真的挂载。

### 2. 补 M 在 web 路径下的可观测性

- **现状**：M 通过 prompt section 注入，在 headless 路径已实测确认（系统提示词从 4034 增至 10363 字符）。但 web 控制台的 Area B **看不到 M 在起作用**——因为 M 是一次性注入，不产生事件。
- **建议**：在 Area B 顶部显示 M 的实时状态（已挂载 / 约 1563 tokens 常驻），数据可从 `run.json` 的 configuration 推导。
- **验证**：选 `m` 配置时能看到 M 的常驻成本；选 `vanilla` 时明确显示「未挂载」。

---

## P1 — 影响结论可信度

### 3. R 的 paraphrase 缺陷（已知，已记录）

- **现状**：BM25 后端。`stretch a box along one axis` **找不到** `fix_deform`；`thermostat to hold a constant temperature` **找不到** `nvt`。两个用例在 `tests/test_retrieval.py` 中是 **strict xfail**。
- **为什么重要**：论文说 R 存在的意义正是「agent 不知道正确术语」的情况。所以**我们测到的 R 贡献是论文 R 的下界**，不能直接对比。
- **修法**：把 83MB 的 ONNX 模型放进 `data/models/`，dense 后端接口已预留。网络能稳定拉下时再做。
- **验证**：两个 xfail 转为通过（strict 模式会在它们通过时**报错**，逼你回来处理）。

### 4. Level 4 的物理判据仍然很薄

- **现状**：所有 task 的 `human_review_required` 项都还挂着（如「晶体是否真的熔化」）。这是**刻意的**——自动判据不足时不猜。
- **建议**：若要减少人工复核，需要引入结构序参量（如 g(r)、Steinhardt 参数）或扩散系数拟合，且必须为每个判据定义清楚的分析窗口。
- **验证**：新增判据后，五个参考实现的 `sanity` 仍应为 100%。

### 5. Web UI 的配置选择器目前是「信息性」的

- **现状**：下拉框可选四配置，但**实际生效的是启动 harness 时挂载的 overlay**。切换需要重启 `start.py --configuration <x>`。
- **原因**：web profile 的 preset roster 解析 harness 自带 preset，不认我们的 roots，传 `agentPreset` 会让 session 创建失败。
- **修法**：让 `start.py` 支持同时挂载四个配置（或按需重启），或改成每次 run 启动独立 harness。
- **验证**：界面上切换配置后，Area B 的顶部说明与工具数量应随之变化。

---

## P2 — 体验与扩展

### 6. 补齐 benchmark 任务的 ground truth 覆盖

- 现状：5 个参考实现都在，且**结构校验 + 真实 LAMMPS 执行双重通过**。
- 可做：为每个 task 增加 1–2 个「写法不同但等价」的变体，用来**证明评分确实不奖励文本相似度**。这是设计原则第 6、8 条的直接检验。

### 7. 前端未做响应式

- 现状：三栏布局在窄屏会挤压。
- 验证：浏览器窗口缩到 900px 以下时三栏仍可读。

### 8. `start.py` 未处理 harness 崩溃后的自动重启

- 现状：harness 退出时会打印退出码并整体关闭（刻意如此——一个所有面板都空的控制台比明确报错更糟）。
- 可做：加 `--restart-on-crash`。

---

## 已确认不需要做的（按设计原则）

self-evolution（论文有，但本项目阶段不要求）· 多智能体 · 自动假设生成 · 材料逆向设计 · 贝叶斯优化 · ML 势函数训练 · DFT 工作流 · 自动写论文 · 自主长周期科研 campaign。

---

## 环境与运维备忘

| 事项 | 说明 |
|---|---|
| 启动 | `.venv/bin/python start.py`（默认 mrsx，Ctrl-C 停止） |
| 端口 | harness 3081 · 控制台 8090 —— 你日常的 GUI 在 **3080，未被触碰** |
| `DSH_HOME` | 隔离在仓库内的 `.dsh-web`，不写 `~/.dsh` |
| `DSH_PERMISSION_MODE` | 必须为 `danger-full-access`：workspace-write sandbox 在本机无法启动，且提权请求无人应答会卡住整轮 |
| SSH 证书 | `sy_hl_login` 的证书 **2026-10-22 到期**，届时需要续期；`hpc/preflight` 会把认证失败单独报出来 |
| 索引重建 | corpus 换了之后：`.venv/bin/python -m adapter.cli build-index` |
| 测试 | `.venv/bin/python -m pytest -q`（当前 300 通过 / 2 xfail / 1 skip） |

---

## 新增：两处本轮发现的问题

### 已排除 — 执行位置其实是对的（先前判断有误，此处更正）

- **我最初的判断**：agent 只在本地跑，没走超算。**这个判断是错的。**
- **实际证据**（`run-980ba3f2e7`）：`run.json` 记录 `job_id 62878459`、`submitted: true`、上传 9 个文件；审计日志里有**三次**提交 `62878374 / 62878378 / 62878459`。
- **三个作业的真实结果**：`62878374` **FAILED**（exit 2:0，0 秒）→ `62878378` **COMPLETED**（11 秒）→ `62878459` **COMPLETED**（8 秒）。
- **结论**：**HPC 路径完全正常工作**，而且 agent 在第一次失败后自己诊断并连续两次提交成功。exit code `2` 正是作业脚本里的输入文件守卫触发的——**它按设计把失败变成了可读的原因**，而不是让 LAMMPS 抛出难懂的错。
- **本地运行的性质**：是提交前的廉价诊断（`run 0` / 短跑），属于合理做法，不是「用错了机器」。
- **仍然保留的改动**：M 里新增的 "Where to run" 一节。它不是修 bug，而是把「运行在集群、本地仅作诊断、提交失败要报告而非静默回退」写成明确政策——**原来的 primer 确实没写，agent 做对了是它自己判断的**。这条不影响正确性，只是让行为可预期。
- **可选验证**：新建任务时确认 agent 仍然正确区分二者（它本来就做对了）。

### P2 — OVITO 可视化（部分可用）

- **现状**：`ovito 3.16.1` 已装入 venv。**但 macOS arm64 的 wheel 有打包 bug**：`ovito_bindings.so` 链接 `libospray.3.2.0.dylib`，而包里只提供 `libospray.3.dylib`。
- **临时修法**：在 `.venv/lib/python3.12/site-packages/ovito/plugins/` 下软链
  `ln -sf libospray.3.dylib libospray.3.2.0.dylib`
  ⚠️ **重装 ovito 后需重做**，建议写进 setup 脚本。
- **未验证**：加载、RDF/配位数计算、离屏渲染**都还没测**（当时的 dump 文件已随临时工作区清理）。
- **下一步**：先在当前工作区跑一个会输出 `dump` 的任务，再用它验证
  - 能否 `import_file` 读取 LAMMPS dump
  - RDF / 配位数 / CNA 能否计算
  - `Viewport.render_image` 在无显示环境能否成功（macOS 上可能失败，若失败则退化为「导出数据 + 只算不渲染」）
