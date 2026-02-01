---
name: report-circt-bug
description: 处理 CIRCT 崩溃用例，执行完整的 Bug 报告流程。使用并行 Agent 调度优化执行效率。主控制流程，协调各子 Skill 完成：复现验证 → 根因分析 → 测例最小化 → 验证 → 重复检查 → 生成报告。
argument-hint: [crash-directory-path]
allowed-tools: Shell(circt-verilog:*), Shell(firtool:*), Shell(gh:*), Shell(cat:*), Shell(ls:*), Shell(mkdir:*), Shell(chmod:*), Shell(jq:*), Read, Write, Grep, Glob, delegate_task, background_output, background_cancel
---

# Skill: 报告 CIRCT Bug (并行调度版)

这是报告 CIRCT Bug 的主控制流程。使用 **并行 Agent 调度** 优化执行效率，通过 `delegate_task` 同时启动多个后台 Agent 处理独立任务。

## 并行工作流架构

```
                         ┌──────────────────────────┐
                         │   report-circt-bug       │
                         │   (Orchestrator Role)    │
                         └────────────┬─────────────┘
                                      │
═══════════════════════════════════════════════════════════════
  Phase 1: 并行初始化 (同时启动 2 个后台 Agent)
═══════════════════════════════════════════════════════════════
                                      │
         ┌────────────────────────────┼────────────────────────┐
         ▼                                                     ▼
  ┌──────────────────┐                            ┌──────────────────────┐
  │ reproduce        │                            │ root-cause-analysis  │
  │ 后台 Agent       │                            │ 后台 Agent           │
  │ 输出: metadata   │                            │ 输出: analysis.json  │
  └──────────────────┘                            └──────────────────────┘
         │                                                     │
         └─────────────────────┬───────────────────────────────┘
                               │ 等待完成 & 检查 reproduce 结果
                               │
═══════════════════════════════════════════════════════════════
  Phase 2: 并行处理 (同时启动 2 个后台 Agent)
═══════════════════════════════════════════════════════════════
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                                           ▼
  ┌────────────────────────┐              ┌──────────────────────┐
  │ minimize → validate    │              │ check-duplicates     │
  │ 后台 Agent (串行执行)   │              │ 后台 Agent           │
  │ 输出: bug.sv, valid.*  │              │ 输出: duplicates.*   │
  └────────────────────────┘              └──────────────────────┘
         │                                           │
         └─────────────────────┬─────────────────────┘
                               │ 等待完成 & 检查 validate 结果
                               │
═══════════════════════════════════════════════════════════════
  Phase 3: 最终报告
═══════════════════════════════════════════════════════════════
                               │
                               ▼
                    ┌──────────────────────┐
                    │ generate-issue       │
                    │ 同步执行             │
                    │ 输出: issue.md       │
                    └──────────────────────┘
                               │
                               ▼
                        report_ready ✅
```

## 最终状态定义

| 状态码 | 描述 | 触发条件 |
|--------|------|----------|
| `not_a_bug` | 不是 Bug | validate 判定为设计限制或无效测例 |
| `duplicate` | 重复 Issue | check-duplicates 发现高度相似的已有 Issue |
| `reproduce_failed` | 复现失败 | reproduce 无法复现崩溃 |
| `report_ready` | 报告就绪 | issue.md 已生成，待人工审核 |
| `submitted` | 已提交 | Issue 已提交到 GitHub |

## 输入要求

输入目录必须包含：
- `source.sv` (或 `.fir`/`.mlir`) - 触发崩溃的测例
- `error.txt` - 崩溃日志

可选但推荐：
- `../circt-src` - CIRCT 源码目录（用于根因分析）

工作目录为当前文件夹。

## 执行流程

### Step 0: 初始化

```bash
# 检查 status.json 是否存在
if [ -f status.json ]; then
    CURRENT_STATUS=$(jq -r '.status' status.json)
    CURRENT_PHASE=$(jq -r '.phase' status.json)
    
    case $CURRENT_STATUS in
        not_a_bug|duplicate|reproduce_failed|report_ready|submitted)
            echo "工作流已完成 ($CURRENT_STATUS)"
            exit 0
            ;;
        processing)
            echo "发现中断的工作流 (phase: $CURRENT_PHASE)，将恢复执行"
            ;;
    esac
fi

# 检查输入文件
SOURCE_FILE=""
[ -f source.sv ] && SOURCE_FILE="source.sv"
[ -f source.fir ] && SOURCE_FILE="source.fir"  
[ -f source.mlir ] && SOURCE_FILE="source.mlir"

if [ -z "$SOURCE_FILE" ] || [ ! -f error.txt ]; then
    echo "Error: Missing required files (source.sv/fir/mlir and error.txt)"
    exit 1
fi

# 初始化 status.json
TIMESTAMP=$(date -Iseconds)
cat > status.json << EOF
{
  "crash_id": "$(basename $(pwd))",
  "status": "processing",
  "phase": "initializing",
  "created_at": "$TIMESTAMP",
  "updated_at": "$TIMESTAMP",
  "parallel_mode": true,
  "input": {
    "source_file": "$SOURCE_FILE",
    "error_file": "error.txt"
  },
  "phases": {
    "phase1": {"status": "pending"},
    "phase2": {"status": "pending"},
    "phase3": {"status": "pending"}
  }
}
EOF
```

### Phase 1: 并行初始化

**同时启动两个后台 Agent:**

```
使用 delegate_task 启动:

1. reproduce-worker (后台)
   - category: "quick"
   - load_skills: []  (使用当前目录的 skill)
   - prompt: 执行 /reproduce skill，报告复现结果
   - run_in_background: true

2. root-cause-worker (后台)
   - category: "unspecified-high" 
   - load_skills: []
   - prompt: 执行 /root-cause-analysis skill，生成 analysis.json
   - run_in_background: true
```

**Worker 1: reproduce-worker 提示模板**

```
你是 reproduce-worker，负责验证 CIRCT Bug 可复现。

工作目录: {当前目录}

任务:
1. 读取 error.txt 提取原始命令
2. 用当前 CIRCT 工具链执行复现
3. 比对崩溃签名确认一致性
4. 生成 reproduce.log 和 metadata.json

输出文件:
- reproduce.log: 复现输出
- metadata.json: 包含 reproduction.reproduced (boolean)

完成后报告:
- reproduced: true/false
- metadata.json 已生成
```

**Worker 2: root-cause-worker 提示模板**

```
你是 root-cause-worker，负责分析 CIRCT 崩溃根因。

工作目录: {当前目录}
CIRCT 源码: ../circt-src (只读，如果存在)

任务:
1. 解析 error.txt 提取错误上下文
2. 分析测例代码识别问题构造
3. 如果 CIRCT 源码可用，定位崩溃点
4. 形成根因假设
5. 生成 root_cause.md 和 analysis.json

输出文件:
- root_cause.md: 详细分析报告
- analysis.json: 结构化分析数据

完成后报告:
- dialect: 识别的方言
- crash_type: 崩溃类型
- analysis.json 已生成
```

**等待并收集结果:**

```
使用 background_output 收集两个 worker 的结果
检查 metadata.json 中的 reproduction.reproduced

IF reproduced == false:
    更新 status.json: status = "reproduce_failed"
    结束流程
ELSE:
    继续 Phase 2
```

### Phase 2: 并行处理

**前置条件检查:**
- `analysis.json` 存在 (Phase 1 输出)
- `reproduce.log` 显示复现成功

**同时启动两个后台 Agent:**

```
使用 delegate_task 启动:

1. minimize-validate-worker (后台)
   - category: "unspecified-high"
   - load_skills: []
   - prompt: 串行执行 /minimize 和 /validate skill
   - run_in_background: true

2. check-duplicates-worker (后台)
   - category: "quick"
   - load_skills: []
   - prompt: 执行 /check-duplicates skill
   - run_in_background: true
```

**Worker 1: minimize-validate-worker 提示模板**

```
你是 minimize-validate-worker，负责最小化测例并验证有效性。

工作目录: {当前目录}
依赖文件: analysis.json, source.sv, error.txt

任务 (串行执行):
1. 执行 /minimize skill
   - 基于 analysis.json 保留关键构造
   - 生成 bug.sv, error.log, command.txt
   
2. 执行 /validate skill
   - 检查 bug.sv 语法有效性
   - 跨工具验证 (Verilator, Slang 如可用)
   - 生成 validation.json, validation.md

输出文件:
- bug.sv: 最小化测例
- error.log: 最小化错误日志
- command.txt: 复现命令
- validation.json: 验证数据
- validation.md: 验证报告

完成后报告:
- validation.result: report/not_a_bug/feature_request/invalid_testcase
- reduction_percent: 最小化比例
```

**Worker 2: check-duplicates-worker 提示模板**

```
你是 check-duplicates-worker，负责检查 GitHub 重复 Issue。

工作目录: {当前目录}
依赖文件: analysis.json

前置条件: gh CLI 已认证

任务:
1. 从 analysis.json 提取关键词
2. 搜索 llvm/circt Issues
3. 计算相似度分数
4. 生成 duplicates.json, duplicates.md

输出文件:
- duplicates.json: 搜索结果和分数
- duplicates.md: 重复检查报告

完成后报告:
- recommendation: new_issue/likely_new/review_existing
- top_score: 最高相似度分数
- top_issue: 最相似的 Issue #
```

**等待并收集结果:**

```
使用 background_output 收集两个 worker 的结果
检查 validation.json 中的 classification.result

IF result IN ["not_a_bug", "invalid_testcase"]:
    更新 status.json: status = "not_a_bug"
    结束流程

IF result == "existing_issue":
    更新 status.json: status = "duplicate"  
    结束流程

检查 duplicates.json 中的 recommendation
IF recommendation == "review_existing" AND top_score >= 10.0:
    更新 status.json: status = "duplicate" (建议人工确认)
    结束流程

继续 Phase 3
```

### Phase 3: 最终报告

**前置条件检查:**
- 所有 Phase 2 输出文件存在
- validate 结果允许继续

**执行 generate-issue (同步):**

```
由主 Agent 直接执行 /generate-issue skill
或使用 delegate_task 同步执行

输入文件:
- bug.sv
- error.log
- command.txt
- root_cause.md
- analysis.json
- validation.md
- duplicates.md
- metadata.json

输出文件:
- issue.md: 完整的 GitHub Issue 报告
```

**完成:**

```bash
# 更新最终状态
jq '.status = "report_ready" | .phase = "completed" | .updated_at = "'"$(date -Iseconds)"'"' status.json > tmp.json && mv tmp.json status.json

echo "✅ 报告生成完成!"
echo "查看: issue.md"
```

## 并行调度代码示例

### Phase 1 并行调度

```
# 启动 Phase 1 的两个后台任务

# Task 1: reproduce
reproduce_task_id = delegate_task(
    category="quick",
    load_skills=[],
    description="reproduce CIRCT crash",
    prompt="""
    工作目录: ./
    执行复现验证:
    1. 读取 error.txt 提取命令
    2. 执行复现
    3. 生成 reproduce.log 和 metadata.json
    4. 报告: reproduced=true/false
    """,
    run_in_background=True
)

# Task 2: root-cause-analysis  
rca_task_id = delegate_task(
    category="unspecified-high",
    load_skills=[],
    description="analyze root cause",
    prompt="""
    工作目录: ./
    执行根因分析:
    1. 分析 error.txt 和测例代码
    2. 如有 ../circt-src，定位崩溃点
    3. 生成 root_cause.md 和 analysis.json
    4. 报告: dialect, crash_type
    """,
    run_in_background=True
)

# 等待两者完成
reproduce_result = background_output(task_id=reproduce_task_id, block=True)
rca_result = background_output(task_id=rca_task_id, block=True)

# 检查 reproduce 结果
metadata = json.load(open('metadata.json'))
if not metadata['reproduction']['reproduced']:
    # 结束流程
    pass
```

### Phase 2 并行调度

```
# 启动 Phase 2 的两个后台任务

# Task 1: minimize + validate (串行)
mv_task_id = delegate_task(
    category="unspecified-high",
    load_skills=[],
    description="minimize and validate",
    prompt="""
    工作目录: ./
    串行执行:
    1. 最小化: 基于 analysis.json 保留关键构造，生成 bug.sv
    2. 验证: 检查 bug.sv 有效性，生成 validation.json
    报告: validation.result, reduction_percent
    """,
    run_in_background=True
)

# Task 2: check-duplicates
dup_task_id = delegate_task(
    category="quick",
    load_skills=[],
    description="check duplicates",
    prompt="""
    工作目录: ./
    检查重复 Issue:
    1. 从 analysis.json 提取关键词
    2. 搜索 llvm/circt Issues
    3. 生成 duplicates.json
    报告: recommendation, top_score
    """,
    run_in_background=True
)

# 等待两者完成
mv_result = background_output(task_id=mv_task_id, block=True)
dup_result = background_output(task_id=dup_task_id, block=True)
```

## 性能对比

| 模式 | Phase 1 | Phase 2 | Phase 3 | 总计 |
|------|---------|---------|---------|------|
| **串行** | 3min + 5min = 8min | 3min + 2min + 1min = 6min | 1min | **15min** |
| **并行** | max(3, 5) = 5min | max(5, 1) = 5min | 1min | **11min** |
| **加速比** | 1.6x | 1.2x | 1x | **1.4x** |

## 上下文优化

| 模式 | 上下文占用 | 说明 |
|------|------------|------|
| 串行 | ~50K tokens | 所有步骤在同一会话累积 |
| 并行 | 主调度 ~5K + workers 各自独立 | 避免上下文膨胀 |

关键优势:
- 每个 worker 只处理自己的任务，不携带历史包袱
- Worker 失败不会污染其他 worker
- 更易于重试和恢复

## 错误处理

### Worker 超时处理

```
默认超时: 10 分钟
超时后:
1. 记录到 status.json
2. 尝试重试一次
3. 如果仍失败，根据 worker 重要性决定:
   - reproduce 失败: 致命，结束
   - root-cause 失败: 非致命，继续但标记
   - minimize 失败: 致命，结束
   - validate 失败: 非致命，假设 report
   - duplicates 失败: 非致命，假设无重复
```

### 恢复策略

```
根据 status.json 中的 phase 恢复:

phase1_parallel:
  检查 metadata.json 和 analysis.json 是否存在
  重跑缺失的 worker

phase2_parallel:
  检查 bug.sv, validation.json, duplicates.json
  重跑缺失的 worker

phase3_final:
  重跑 generate-issue
```

## 输出文件结构

```
circt-b<id>/
├── status.json              # 状态跟踪 (含并行状态)
├── source.sv                # 原始测例 (输入)
├── error.txt                # 原始日志 (输入)
│
├── # Phase 1 输出
├── reproduce.log            # 复现输出
├── metadata.json            # 复现元数据
├── root_cause.md            # 根因分析报告
├── analysis.json            # 根因分析数据
│
├── # Phase 2 输出
├── bug.sv                   # 最小化测例
├── error.log                # 最小化日志
├── command.txt              # 复现命令
├── minimize_report.md       # 最小化报告
├── validation.json          # 验证数据
├── validation.md            # 验证报告
├── duplicates.json          # 重复检查数据
├── duplicates.md            # 重复检查报告
│
├── # Phase 3 输出
└── issue.md                 # 最终 Issue 报告
```

## 注意事项

1. **CIRCT 工具**: 确保 `circt-verilog` 或 `firtool` 在 PATH 中
2. **GitHub CLI**: `gh` CLI 需要认证 (用于重复检查)
3. **CIRCT 源码**: `../circt-src` 用于根因分析 (可选)
4. **并行安全**: Worker 不会同时写入同一文件
5. **状态跟踪**: status.json 记录并行任务状态
6. **中断恢复**: 支持从任意 phase 恢复

## 调试提示

```bash
# 查看当前状态
cat status.json | jq '.status, .phase'

# 查看并行任务状态
cat status.json | jq '.phases'

# 强制重新开始
rm status.json && # 调用 /report-circt-bug

# 只重跑某个 phase
jq '.phase = "phase1_parallel"' status.json > tmp.json && mv tmp.json status.json
```
