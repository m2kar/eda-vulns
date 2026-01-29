---
name: root-cause-analysis
description: AI 驱动的 CIRCT 崩溃根因分析。结合错误日志、测例代码和 CIRCT 源码进行深度分析，生成详细的根因报告和结构化分析数据。
argument-hint: (无参数，在当前目录操作)
allowed-tools: Read, Write, Grep, Glob, Shell(cat:*), Shell(ls:*), Shell(head:*), Shell(tail:*)
---

# Skill: 根因分析 (AI-Powered)

## 功能描述

使用 AI 推理能力进行深度根因分析，结合：
1. 错误日志（stack trace, assertion messages）
2. 测例代码（SystemVerilog/FIRRTL/MLIR）
3. CIRCT 源码（定位崩溃点，理解处理逻辑）

生成详细的根因报告和结构化分析数据。

## 输入

当前目录必须包含：
- `error.txt` - 崩溃日志
- `source.sv` (或 `.fir`/`.mlir`) - 测例代码
- `metadata.json` - 复现元数据（来自 reproduce skill）

CIRCT 源码位于：`../circt-src`（只读）

## 输出

- `root_cause.md` - 详细根因分析报告
- `analysis.json` - 结构化分析数据

## 分析流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    ROOT CAUSE ANALYSIS                          │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: Parse Error Context                                    │
│  ├── Extract assertion message                                  │
│  ├── Extract stack trace                                        │
│  ├── Identify failing pass/dialect                              │
│  └── Extract source file:line from crash                        │
├─────────────────────────────────────────────────────────────────┤
│  Step 2: Analyze Test Case                                      │
│  ├── Identify language (SV/FIRRTL/MLIR)                         │
│  ├── Identify key constructs used                               │
│  ├── Find potentially problematic patterns                      │
│  └── Understand test intent                                     │
├─────────────────────────────────────────────────────────────────┤
│  Step 3: Explore CIRCT Source Code                              │
│  ├── Locate crash site in source (../circt-src)                 │
│  ├── Read surrounding code context                              │
│  ├── Trace call path from stack frames                          │
│  └── Understand the failing logic                               │
├─────────────────────────────────────────────────────────────────┤
│  Step 4: Correlate and Reason                                   │
│  ├── Map test constructs to compiler handling                   │
│  ├── Identify gap between expected and actual behavior          │
│  ├── Form hypotheses about root cause                           │
│  └── Validate hypotheses against evidence                       │
├─────────────────────────────────────────────────────────────────┤
│  Step 5: Generate Report                                        │
│  ├── Executive summary                                          │
│  ├── Technical deep dive                                        │
│  ├── Ranked hypotheses with evidence                            │
│  └── Suggested fix directions                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Step 1: Parse Error Context

从 `error.txt` 提取关键信息：

```bash
# 提取 assertion message
ASSERTION=$(grep -E 'Assertion.*failed' error.txt | head -1)

# 提取 LLVM ERROR
LLVM_ERROR=$(grep -E 'LLVM ERROR:' error.txt | head -1)

# 提取 stack trace（过滤 circt/mlir 相关帧）
grep -E '(circt::|mlir::|llvm::)' error.txt | head -20

# 提取崩溃位置
CRASH_LOCATION=$(grep -oE '[A-Za-z]+\.cpp:[0-9]+' error.txt | head -1)
```

**关键模式识别**：

| 模式 | 含义 |
|------|------|
| `dyn_cast on a non-existent value` | MLIR value 空指针 |
| `failed to legalize operation` | Pass 无法转换操作 |
| `use of value.*different block` | SSA dominance 违规 |
| `Assertion.*failed` | 内部不变量被破坏 |
| `llvm_unreachable` | 到达了不应到达的代码路径 |
| `unhandled case` | switch/match 缺少处理器 |

## Step 2: Analyze Test Case

读取测例代码，识别：

```
分析测例代码：
1. 模块结构和层次
2. 关键语言构造：
   - 数据类型（union, struct, enum, array）
   - 过程块（always_ff, always_comb, initial）
   - 控制流（if/else, case, for/while）
   - 接口、包、类
   - 断言、覆盖组
3. 潜在问题组合：
   - 敏感列表中的数组索引
   - 嵌套的 packed union/struct
   - 复杂参数化
   - 不支持的 SV 特性（DPI, classes 等）
```

## Step 3: Explore CIRCT Source Code

**CIRCT 源码位于**: `../circt-src`（只读）

根据崩溃上下文导航：

```
CIRCT 源码目录结构：

lib/Conversion/MooreToCore/     - SV 到核心方言的转换
lib/Dialect/Moore/              - Moore 方言定义
lib/Dialect/FIRRTL/Transforms/  - FIRRTL pass 实现
lib/Conversion/FIRRTLToHW/      - FIRRTL 到 HW 的转换
lib/Conversion/ExportVerilog/   - HW 到 Verilog 输出
include/circt/Dialect/*/        - 方言 TableGen 定义
```

**导航步骤**：
1. 从 stack trace 定位崩溃文件:行号
2. 读取包含崩溃的函数
3. 理解上下文和预期的不变量
4. 追踪处理路径

## Step 4: Correlate and Reason

应用推理连接观察：

```
推理框架：

1. 输入 → 处理 → 崩溃链：
   - 什么具体的输入模式触发了这个？
   - 哪个 pass/函数处理这个模式？
   - 为什么处理失败？

2. 预期 vs 实际：
   - 编译器应该对这个输入做什么？
   - 它实际在做什么？
   - 分歧在哪里发生？

3. 假设形成：
   - H1: [最可能的原因] - [证据]
   - H2: [替代原因] - [证据]
   - H3: [较不可能的原因] - [证据]

4. 验证：
   - 假设能解释所有症状吗？
   - 有矛盾的观察吗？
   - 什么可以确认/反驳假设？
```

## Step 5: Generate Report

### root_cause.md 格式

```markdown
# Root Cause Analysis Report

## Executive Summary
[2-3 句话总结 Bug 和可能的原因]

## Crash Context
- **Tool/Command**: [circt-verilog, firtool, etc.]
- **Dialect**: [Moore, FIRRTL, HW, etc.]
- **Failing Pass**: [pass name if identified]
- **Crash Type**: [Assertion, Segfault, etc.]

## Error Analysis

### Assertion/Error Message
```
[exact assertion or error message]
```

### Key Stack Frames
```
[relevant stack frames, filtered for circt/mlir]
```

## Test Case Analysis

### Code Summary
[测例做什么的简要描述]

### Key Constructs
- [construct 1]: [与崩溃的关系]
- [construct 2]: [与崩溃的关系]

### Potentially Problematic Patterns
[可能导致问题的具体模式]

## CIRCT Source Analysis

### Crash Location
**File**: [filename.cpp]
**Function**: [function name]
**Line**: [approximate line]

### Code Context
```cpp
[relevant code snippet from CIRCT source]
```

### Processing Path
1. [处理的第一步]
2. [第二步]
3. [失败的地方和原因]

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: [描述]
**Evidence**:
- [证据点 1]
- [证据点 2]
**Mechanism**: [这如何导致崩溃]

### Hypothesis 2 (Medium Confidence)
[类似结构]

## Suggested Fix Directions
1. [建议 1 及理由]
2. [建议 2 及理由]

## Keywords for Issue Search
`keyword1` `keyword2` `keyword3` ...

## Related Files to Investigate
- `path/to/file1.cpp` - [原因]
- `path/to/file2.cpp` - [原因]
```

### analysis.json 格式

```json
{
  "version": "2.0",
  "analysis_type": "ai_reasoning",
  "dialect": "Moore",
  "failing_pass": "MooreToCore",
  "crash_type": "assertion",
  "assertion_message": "...",
  "crash_location": {
    "file": "MooreToCore.cpp",
    "function": "...",
    "line": 123
  },
  "test_case": {
    "language": "systemverilog",
    "key_constructs": ["packed union", "array indexing"],
    "problematic_patterns": ["array in sensitivity list"]
  },
  "hypotheses": [
    {
      "description": "...",
      "confidence": "high",
      "evidence": ["...", "..."]
    }
  ],
  "keywords": ["keyword1", "keyword2"],
  "suggested_sources": [
    {"path": "...", "reason": "..."}
  ]
}
```

## 崩溃类型识别

| 模式 | 可能的 Dialect/工具 |
|------|---------------------|
| `MooreToCore` | Moore (circt-verilog) |
| `firrtl::` | FIRRTL (firtool) |
| `arc::` | Arc (arcilator) |
| `hw::` | HW dialect |
| `seq::` | Seq dialect |
| `sv::` | SV dialect |
| `comb::` | Comb dialect |
| `llhd::` | LLHD dialect |

## 崩溃类别

| 类别 | 描述 |
|------|------|
| Null/Invalid Value Access | dyn_cast 在不存在的值上 |
| Legalization Failure | 无法转换操作 |
| SSA Violation | 值在定义范围外使用 |
| Type Mismatch | 类型预期不满足 |
| Incomplete Implementation | 未处理的情况 |
| Assertion Failure | 内部不变量被破坏 |

## 质量检查清单

完成分析前确认：
- [ ] 错误上下文完整提取（assertion, stack trace）
- [ ] 测例构造已识别
- [ ] 至少检查了一个 CIRCT 源文件
- [ ] 假设有支持证据
- [ ] 关键词可用于 Issue 搜索
- [ ] 报告对开发者可操作

## 注意事项

1. **CIRCT 源码只读**：`../circt-src` 不要修改
2. **深度推理**：AI 应该主动阅读源码，而不仅仅是模式匹配
3. **证据驱动**：每个假设必须有具体证据支持
4. **关键词质量**：生成的关键词应该能有效用于 Issue 搜索
5. **可操作性**：报告应该帮助开发者快速理解和修复问题
