# CIRCT Bug 报告技能文件

本目录包含用于在单个 `circt-b<id>` 工作目录中处理 CIRCT Bug 报告任务的技能文件。

## 工作流架构 (并行版)

采用 **并行 Agent 调度** 架构，通过 `delegate_task` 同时启动多个后台 Agent 处理独立任务，实现 **1.4x 加速**。

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

### 性能对比

| 模式 | Phase 1 | Phase 2 | Phase 3 | 总计 |
|------|---------|---------|---------|------|
| **串行** | 3min + 5min = 8min | 3min + 2min + 1min = 6min | 1min | **15min** |
| **并行** | max(3, 5) = 5min | max(5, 1) = 5min | 1min | **11min** |
| **加速比** | 1.6x | 1.2x | 1x | **1.4x** |

## 技能列表

### 主技能 (Orchestrator)

| 技能 | 描述 |
|------|------|
| **report-circt-bug** | 主控制流程 (Orchestrator Role)，使用 `delegate_task` 并行调度各 Worker Agent |

### Worker 技能

| 技能 | Phase | 执行方式 | 输出文件 |
|------|-------|----------|----------|
| **reproduce** | 1 | 后台 Agent | `reproduce.log`, `metadata.json` |
| **root-cause-analysis** | 1 | 后台 Agent | `root_cause.md`, `analysis.json` |
| **minimize** | 2 | 后台 Agent (串行) | `bug.sv`, `error.log`, `command.txt` |
| **validate** | 2 | 后台 Agent (串行) | `validation.json`, `validation.md` |
| **check-duplicates** | 2 | 后台 Agent | `duplicates.json`, `duplicates.md` |
| **generate-issue** | 3 | 同步执行 | `issue.md` |
| **check-status** | - | 辅助工具 | 更新 `status.json` |

### 架构文档

| 文档 | 位置 | 描述 |
|------|------|------|
| **Orchestrator Agent** | `.claude/agents/orchestrator.md` | 主调度器设计文档，定义并行执行策略 |

## 状态管理

### status.json 文件

每个 circt-b<id> 目录中都有一个 `status.json` 文件，用于跟踪工作流状态。

#### 最终状态

| 状态码 | 描述 | 何时设置 |
|--------|------|----------|
| `not_a_bug` | 不是 Bug | validate 判定为设计限制或无效测例 |
| `duplicate` | 重复 Issue | check-duplicates 发现高相似度 Issue |
| `reproduce_failed` | 复现失败 | reproduce 无法复现崩溃 |
| `report_ready` | 报告就绪 | issue.md 已生成，待人工审核 |
| `submitted` | 已提交 | Issue 已提交到 GitHub |

#### 中间状态

| 状态码 | 描述 | 对应阶段 |
|--------|------|----------|
| `pending` | 待处理 | 初始状态 |
| `reproducing` | 复现中 | 正在验证复现 |
| `analyzing` | 分析中 | 正在进行根因分析 |
| `minimizing` | 最小化中 | 正在最小化测例 |
| `validating` | 验证中 | 正在验证测例 |
| `checking_duplicates` | 检查重复 | 正在检查重复 Issue |
| `generating` | 生成中 | 正在生成报告 |

#### JSON Schema

```json
{
  "crash_id": "circt-b1",
  "status": "report_ready",
  "phase": "completed",
  "created_at": "2026-01-27T14:00:00Z",
  "updated_at": "2026-01-27T16:00:00Z",
  "input": {
    "source_dir": "/path/to/crash/directory",
    "error_file": "error.txt",
    "source_file": "source.sv"
  },
  "steps": {
    "reproduce": {
      "completed": true,
      "timestamp": "2026-01-27T14:05:00Z",
      "reproduced": true,
      "tool_version": "firtool-1.140.0",
      "files": ["reproduce.log"]
    },
    "root_cause_analysis": {
      "completed": true,
      "timestamp": "2026-01-27T14:15:00Z",
      "dialect": "Moore",
      "failing_pass": "MooreToCore",
      "crash_type": "assertion",
      "files": ["root_cause.md", "analysis.json"]
    },
    "minimize": {
      "completed": true,
      "timestamp": "2026-01-27T14:30:00Z",
      "original_lines": 156,
      "minimized_lines": 42,
      "reduction_percent": 73.1,
      "files": ["bug.sv", "error.log", "command.txt"]
    },
    "validate": {
      "completed": true,
      "timestamp": "2026-01-27T14:40:00Z",
      "result": "report",
      "files": ["validation.json", "validation.md"]
    },
    "check_duplicates": {
      "completed": true,
      "timestamp": "2026-01-27T14:50:00Z",
      "similar_count": 2,
      "recommendation": "likely_new",
      "files": ["duplicates.json", "duplicates.md"]
    },
    "generate_issue": {
      "completed": true,
      "timestamp": "2026-01-27T15:00:00Z",
      "files": ["issue.md"]
    }
  },
  "summary": "Moore dialect assertion failure in MooreToCore pass",
  "error": null
}
```

## 设计原则

### 1. 并行优先
- **Phase 1**: reproduce 和 root-cause-analysis 并行执行
- **Phase 2**: minimize→validate 和 check-duplicates 并行执行
- 通过 `delegate_task(run_in_background=True)` 启动后台 Agent
- 使用 `background_output(task_id, block=True)` 收集结果

### 2. Worker 独立
- 每个 Worker Agent 只处理自己的任务
- 通过文件交换数据（JSON + MD），不传递上下文
- Worker 失败不会污染其他 Worker
- 低耦合、高内聚

### 3. 状态驱动
- `status.json` 作为核心状态管理
- 支持中断恢复（从任意 phase 恢复）
- 可追溯的执行历史

### 4. 文件驱动
- 所有 Agent 间通信通过文件
- 文件即文档
- 便于版本控制和回溯

## 目录结构

```
circt-b<id>/
├── .claude/
│   ├── agents/
│   │   └── orchestrator.md          # Orchestrator Agent 设计文档
│   └── skills/
│       ├── README.md                    # 本文件
│       ├── report-circt-bug/SKILL.md    # 主控制流程 (Orchestrator)
│       ├── reproduce/SKILL.md           # Worker: 复现验证
│       ├── root-cause-analysis/SKILL.md # Worker: 根因分析
│       ├── minimize/SKILL.md            # Worker: 测例最小化
│       ├── validate/SKILL.md            # Worker: 测例验证
│       ├── check-duplicates/SKILL.md    # Worker: 重复检查
│       ├── generate-issue/SKILL.md      # Worker: 生成报告
│       └── check-status/SKILL.md        # 辅助: 状态检查
├── status.json              # 状态跟踪 (含并行任务状态)
├── source.sv                # 原始测例（输入）
├── error.txt                # 原始错误日志（输入）
│
├── # Phase 1 输出
├── reproduce.log            # 复现输出
├── metadata.json            # 复现元数据
├── root_cause.md            # 根因分析报告
├── analysis.json            # 根因分析数据
│
├── # Phase 2 输出
├── bug.sv                   # 最小化测例
├── error.log                # 最小化错误日志
├── command.txt              # 复现命令
├── validation.json          # 验证数据
├── validation.md            # 验证报告
├── duplicates.json          # 重复检查数据
├── duplicates.md            # 重复检查报告
│
└── # Phase 3 输出
    └── issue.md             # 最终 Issue 报告
```

## 使用方法

### 基本用法

```bash
# 1. 进入或创建工作目录
mkdir -p circt-b<id>
cd circt-b<id>

# 2. 准备输入文件
cp /path/to/crash/error.txt .
cp /path/to/crash/source.sv .

# 3. 调用主技能
# 输入：处理当前目录的 CIRCT Bug
# Orchestrator 会自动并行调度各 Worker Agent

# 4. 并行执行流程:
# Phase 1: reproduce || root-cause-analysis
# Phase 2: (minimize → validate) || check-duplicates  
# Phase 3: generate-issue

# 5. 查看结果
cat status.json | jq '.status'  # 查看状态
cat issue.md                     # 查看生成的报告
```

### 中断恢复

```bash
# 查看当前状态
cat status.json | jq '.status'

# 如果是中间状态，继续执行
# 调用 report-circt-bug 会自动从中断点恢复
```

### 单独调用子技能

```bash
# 只运行根因分析
# 调用 /root-cause-analysis skill

# 只运行最小化
# 调用 /minimize skill

# 检查状态
# 调用 /check-status skill
```

## CIRCT 工具参考

| 工具 | 用途 | 常用参数 |
|------|------|----------|
| `circt-verilog` | SV → MLIR | `--ir-hw`, `--ir-moore` |
| `firtool` | FIRRTL 编译器 | `--verilog`, `-O0` ~ `-O3` |
| `circt-opt` | MLIR 优化器 | Pass pipelines |
| `arcilator` | Arc 仿真器 | State lowering |

## 崩溃类型识别

| 模式 | 可能的 Dialect/工具 |
|------|---------------------|
| `MooreToCore` | Moore (circt-verilog) |
| `firrtl::` | FIRRTL (firtool) |
| `arc::` | Arc (arcilator) |
| `hw::` | HW dialect |
| `seq::` | Seq dialect |

## 常见问题

### Q: 如何强制重新开始？
A: 删除 `status.json` 文件后重新执行。

### Q: 工作流中断后如何继续？
A: 直接调用 report-circt-bug，会自动从 status.json 中恢复。

### Q: 为什么使用并行 Agent 调度？
A: 并行调度带来多重优势：
   - **性能**: 独立任务并行执行，1.4x 加速
   - **上下文隔离**: 每个 Worker 只处理自己的任务，避免上下文膨胀
   - **故障隔离**: Worker 失败不会污染其他 Worker
   - **可恢复性**: 支持从任意 phase 恢复

### Q: CIRCT 源码在哪里？
A: CIRCT 源码位于 `../circt-src`（只读），用于根因分析时参考。

## 并行调度详情

详细的并行调度策略、Worker 提示模板、错误处理和恢复策略，请参阅：
- **Orchestrator 设计文档**: `.claude/agents/orchestrator.md`
- **主控 Skill**: `.claude/skills/report-circt-bug/SKILL.md`

## 注意事项

1. **环境变量**：设置 `CIRCT_BIN` 指向 CIRCT 工具目录（如未在 PATH 中）
2. **GitHub CLI**：`gh` CLI 需要已安装并认证（用于重复检查和提交）
3. **单目录操作**：每次只处理一个崩溃用例
4. **状态持久化**：通过 status.json 跟踪进度
5. **中断安全**：支持从任意中间状态恢复
6. **CIRCT 源码**：使用 `../circt-src` 进行根因分析
7. **并行安全**：多个 Worker 不会同时写入同一文件
