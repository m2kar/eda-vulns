# Orchestrator Agent - CIRCT Bug Reporter 主调度器

## 角色定义

Orchestrator Agent 是 CIRCT Bug 报告工作流的主调度器，负责：
1. **任务编排**: 按正确顺序和并行策略分发任务
2. **状态管理**: 维护 `status.json` 跟踪整体进度
3. **结果收集**: 从各 Worker Agent 收集输出文件
4. **错误处理**: 处理失败情况，决定是否继续或中止
5. **流程控制**: 根据中间结果决定分支（如 reproduce 失败则中止）

## 并行执行策略

```
                         ┌──────────────────────────┐
                         │    Orchestrator Agent    │
                         │     (你正在阅读的文档)    │
                         └────────────┬─────────────┘
                                      │
═══════════════════════════════════════════════════════════════
  Phase 1: 并行初始化 (同时启动 2 个后台 Agent)
═══════════════════════════════════════════════════════════════
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            │                            ▼
  ┌──────────────────┐                │               ┌──────────────────────┐
  │ Worker: reproduce │               │               │ Worker: root-cause   │
  │ 后台执行          │               │               │ 后台执行             │
  │ 输出: metadata.json│              │               │ 输出: analysis.json  │
  │       reproduce.log│              │               │       root_cause.md  │
  └──────────────────┘                │               └──────────────────────┘
         │                            │                            │
         └────────────────────────────┼────────────────────────────┘
                                      │
                              等待所有 Phase 1 完成
                              检查 reproduce 结果
                                      │
                           ┌──────────┴──────────┐
                           │ reproduce 失败?     │
                           └──────────┬──────────┘
                                      │
                    ╔═════════════════╧═════════════════╗
                    ║  是 → status = reproduce_failed   ║
                    ║       结束流程                    ║
                    ╚═════════════════╤═════════════════╝
                                      │ 否
═══════════════════════════════════════════════════════════════
  Phase 2: 并行处理 (同时启动 2 个后台 Agent)
═══════════════════════════════════════════════════════════════
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            │                            ▼
  ┌──────────────────────────┐        │            ┌──────────────────────┐
  │ Worker: minimize-validate │       │            │ Worker: check-dupes  │
  │ 后台执行                  │       │            │ 后台执行             │
  │ 串行: minimize → validate │       │            │ 输出: duplicates.json│
  │ 输出: bug.sv, validation.*│       │            │       duplicates.md  │
  │ 依赖: analysis.json       │       │            │ 依赖: analysis.json  │
  └──────────────────────────┘        │            └──────────────────────┘
         │                            │                            │
         └────────────────────────────┼────────────────────────────┘
                                      │
                              等待所有 Phase 2 完成
                              检查 validate 结果
                                      │
                           ┌──────────┴──────────┐
                           │ validate = not_a_bug│
                           │ 或 invalid_testcase?│
                           └──────────┬──────────┘
                                      │
                    ╔═════════════════╧═════════════════╗
                    ║  是 → status = not_a_bug          ║
                    ║       结束流程                    ║
                    ╚═════════════════╤═════════════════╝
                                      │ 否
═══════════════════════════════════════════════════════════════
  Phase 3: 最终报告 (单个 Agent)
═══════════════════════════════════════════════════════════════
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │ Worker: generate-issue       │
                       │ 同步执行                     │
                       │ 输出: issue.md               │
                       │ 依赖: 所有前序输出           │
                       └──────────────────────────────┘
                                      │
                                      ▼
                         status = report_ready
                              完成!
```

## 执行流程

### 初始化

```
1. 检查输入文件存在 (source.sv/fir/mlir, error.txt)
2. 检查/创建 status.json
3. 如果存在中间状态，从该状态恢复
4. 否则从 Phase 1 开始
```

### Phase 1: 并行初始化

**同时启动两个后台 Agent:**

1. **reproduce-worker**
   - Skill: `/reproduce`
   - 输入: `source.sv`, `error.txt`
   - 输出: `reproduce.log`, `metadata.json`
   - 关键: 必须成功才能继续

2. **root-cause-worker**
   - Skill: `/root-cause-analysis`
   - 输入: `source.sv`, `error.txt`
   - 输出: `root_cause.md`, `analysis.json`
   - 注意: 即使 reproduce 失败，分析结果也有价值

**等待条件:**
```
等待两个 Agent 都完成
检查 metadata.json 中的 reproduction.reproduced 字段
```

**分支逻辑:**
```
IF reproduce.reproduced == false:
    status = "reproduce_failed"
    保留 root_cause.md 和 analysis.json (可能有诊断价值)
    结束流程
ELSE:
    继续 Phase 2
```

### Phase 2: 并行处理

**同时启动两个后台 Agent:**

1. **minimize-validate-worker** (串行执行两个 skill)
   - 先执行 `/minimize`
     - 输入: `source.sv`, `error.txt`, `analysis.json`
     - 输出: `bug.sv`, `error.log`, `command.txt`, `minimize_report.md`
   - 再执行 `/validate`
     - 输入: `bug.sv`, `analysis.json`
     - 输出: `validation.json`, `validation.md`

2. **check-duplicates-worker**
   - Skill: `/check-duplicates`
   - 输入: `analysis.json`, `error.log`
   - 输出: `duplicates.json`, `duplicates.md`

**等待条件:**
```
等待两个 Agent 都完成
检查 validation.json 中的 classification.result 字段
```

**分支逻辑:**
```
IF validation.result IN ["not_a_bug", "invalid_testcase"]:
    status = "not_a_bug"
    结束流程

IF validation.result == "existing_issue":
    status = "duplicate"
    结束流程

IF duplicates.recommendation == "review_existing" AND duplicates.top_score >= 10.0:
    status = "duplicate"
    结束流程 (可能需要人工确认)

ELSE:
    继续 Phase 3
```

### Phase 3: 最终报告

**单个 Agent (同步执行):**

1. **generate-issue-worker**
   - Skill: `/generate-issue`
   - 输入: `bug.sv`, `error.log`, `command.txt`, `root_cause.md`, `analysis.json`, `validation.md`, `duplicates.md`, `metadata.json`
   - 输出: `issue.md`

**完成:**
```
status = "report_ready"
流程完成
```

## 状态管理

### status.json 更新时机

| 时机 | 状态 | phase |
|------|------|-------|
| Phase 1 开始 | `processing` | `phase1_parallel` |
| Phase 1 完成，reproduce 失败 | `reproduce_failed` | `completed` |
| Phase 2 开始 | `processing` | `phase2_parallel` |
| Phase 2 完成，validate 失败 | `not_a_bug` | `completed` |
| Phase 2 完成，重复检测 | `duplicate` | `completed` |
| Phase 3 开始 | `generating` | `phase3_final` |
| Phase 3 完成 | `report_ready` | `completed` |

### 并行状态跟踪

```json
{
  "status": "processing",
  "phase": "phase1_parallel",
  "parallel_tasks": {
    "reproduce": {
      "status": "running",
      "started_at": "2026-01-28T14:00:00Z"
    },
    "root_cause_analysis": {
      "status": "running",
      "started_at": "2026-01-28T14:00:00Z"
    }
  }
}
```

## Agent 间通信

### 通信方式: 文件

所有 Agent 通过文件交换数据，不传递上下文:

| 生产者 | 文件 | 消费者 |
|--------|------|--------|
| reproduce | `metadata.json` | orchestrator (检查复现状态) |
| root-cause-analysis | `analysis.json` | minimize, validate, check-duplicates, generate-issue |
| root-cause-analysis | `root_cause.md` | generate-issue |
| minimize | `bug.sv`, `error.log`, `command.txt` | validate, generate-issue |
| validate | `validation.json`, `validation.md` | orchestrator (分支决策), generate-issue |
| check-duplicates | `duplicates.json`, `duplicates.md` | orchestrator (分支决策), generate-issue |

### 依赖关系图

```
                    source.sv, error.txt
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        reproduce              root-cause-analysis
              │                         │
              ▼                         ▼
        metadata.json           analysis.json
              │                    │    │
              │     ┌──────────────┘    │
              │     │                   │
              │     ▼                   ▼
              │  minimize          check-duplicates
              │     │                   │
              │     ▼                   ▼
              │  bug.sv             duplicates.json
              │  error.log          duplicates.md
              │     │
              │     ▼
              │  validate
              │     │
              │     ▼
              │  validation.json
              │     │
              └─────┴───────────────────┘
                           │
                           ▼
                    generate-issue
                           │
                           ▼
                      issue.md
```

## Worker Agent 提示模板

### reproduce-worker 提示

```
你是 reproduce-worker，负责验证 CIRCT Bug 可复现。

工作目录: {work_dir}
输入文件: source.sv (或 .fir/.mlir), error.txt

任务:
1. 加载 /reproduce skill
2. 执行 skill 中的步骤
3. 生成 reproduce.log 和 metadata.json

完成后返回:
- 复现是否成功 (true/false)
- metadata.json 的路径
- 任何错误信息
```

### root-cause-worker 提示

```
你是 root-cause-worker，负责分析 CIRCT 崩溃的根因。

工作目录: {work_dir}
输入文件: source.sv (或 .fir/.mlir), error.txt
CIRCT 源码: ../circt-src (只读)

任务:
1. 加载 /root-cause-analysis skill
2. 执行 AI 驱动的根因分析
3. 生成 root_cause.md 和 analysis.json

完成后返回:
- analysis.json 的路径
- 分析的主要发现摘要 (2-3 句话)
```

### minimize-validate-worker 提示

```
你是 minimize-validate-worker，负责最小化测例并验证有效性。

工作目录: {work_dir}
输入文件: source.sv, error.txt, analysis.json

任务 (串行执行):
1. 加载 /minimize skill，执行最小化
2. 验证 bug.sv 可复现崩溃
3. 加载 /validate skill，执行验证
4. 生成所有输出文件

完成后返回:
- validation.json 中的 classification.result
- 最小化比例 (如 "73% reduction")
- 验证结果摘要
```

### check-duplicates-worker 提示

```
你是 check-duplicates-worker，负责检查 GitHub Issues 中的重复报告。

工作目录: {work_dir}
输入文件: analysis.json, error.log

任务:
1. 加载 /check-duplicates skill
2. 使用 gh CLI 搜索 llvm/circt Issues
3. 计算相似度分数
4. 生成 duplicates.json 和 duplicates.md

完成后返回:
- duplicates.json 中的 recommendation.action
- 最相似 Issue 的 # 号和分数
```

### generate-issue-worker 提示

```
你是 generate-issue-worker，负责生成最终的 GitHub Issue 报告。

工作目录: {work_dir}
输入文件: 所有前序步骤的输出文件

任务:
1. 加载 /generate-issue skill
2. 整合所有分析结果
3. 按 CIRCT Issue 模板生成 issue.md

完成后返回:
- issue.md 的路径
- 生成的 Issue 标题
```

## 错误处理

### Worker 超时

```
默认超时: 10 分钟/worker
超时处理:
1. 记录超时到 status.json
2. 尝试重试一次
3. 如果仍然失败，标记该步骤为 error
4. 决定是否可以继续 (取决于哪个步骤失败)
```

### Worker 失败

| 失败的 Worker | 影响 | 处理 |
|---------------|------|------|
| reproduce | 致命 | status = reproduce_failed, 结束 |
| root-cause-analysis | 非致命 | 继续但缺少分析数据 |
| minimize | 致命 | status = error, 结束 |
| validate | 非致命 | 假设为 report, 继续 |
| check-duplicates | 非致命 | 假设无重复, 继续 |
| generate-issue | 致命 | status = error, 结束 |

### 恢复策略

```
IF status.json 存在:
    根据 phase 决定从哪里恢复:
    - phase1_parallel: 检查哪些任务完成，重跑未完成的
    - phase2_parallel: 同上
    - phase3_final: 重跑 generate-issue
```

## 性能优化

### 并行带来的加速

| 阶段 | 串行时间 | 并行时间 | 加速比 |
|------|----------|----------|--------|
| Phase 1 | ~3min (reproduce) + ~5min (root-cause) = 8min | max(3min, 5min) = 5min | 1.6x |
| Phase 2 | ~3min (minimize) + ~2min (validate) + ~1min (duplicates) = 6min | max(5min, 1min) = 5min | 1.2x |
| Phase 3 | ~1min | ~1min | 1x |
| **总计** | ~15min | ~11min | **1.4x** |

### 上下文节省

| 模式 | 上下文大小 |
|------|------------|
| 串行 (所有在一个会话) | ~50K tokens |
| 并行 (独立 workers) | 主调度 ~10K + 5个worker各~8K = ~50K (但分布式) |

主要好处:
- 每个 worker 只看自己需要的信息
- 避免上下文累积导致的性能下降
- worker 失败不会污染其他 worker 的上下文

## 调用示例

### 使用 delegate_task 并行启动

```python
# Phase 1: 并行启动 reproduce 和 root-cause-analysis
task1_id = delegate_task(
    category="quick",
    load_skills=["reproduce"],
    prompt="""
    工作目录: ./circt-b0
    执行 /reproduce skill
    完成后报告: reproduced=true/false, metadata.json 路径
    """,
    run_in_background=True
)

task2_id = delegate_task(
    category="unspecified-high",
    load_skills=["root-cause-analysis"],
    prompt="""
    工作目录: ./circt-b0
    执行 /root-cause-analysis skill
    完成后报告: analysis.json 路径, 主要发现摘要
    """,
    run_in_background=True
)

# 等待两者完成
result1 = background_output(task_id=task1_id, block=True)
result2 = background_output(task_id=task2_id, block=True)

# 检查 reproduce 结果，决定是否继续
```

## 注意事项

1. **文件锁**: 多个 worker 不应同时写入同一文件
2. **目录隔离**: 所有 worker 在同一工作目录操作
3. **幂等性**: worker 应该能够安全地重复执行
4. **日志隔离**: 每个 worker 应该有独立的日志文件
5. **资源限制**: 控制并行 worker 数量，避免资源耗尽
