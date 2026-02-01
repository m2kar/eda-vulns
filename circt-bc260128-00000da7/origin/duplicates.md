# GitHub Duplicate Issue Check Report

**Test Case ID**: 260128-00000da7  
**Analysis Date**: 2025-02-01  
**Report Generated**: Check Duplicates Skill

---

## 1. 问题摘要

### 原始崩溃信息
- **方言**: Verilog/SystemVerilog
- **崩溃类型**: timeout (timeout 300s)
- **可疑组件**: arcilator
- **可疑优化通道**: ConvertToArcs, LowerState, SplitLoops

### 根本原因假设
Arcilator enters an infinite loop when processing always_comb blocks with dynamic-indexed array elements that have write-then-read patterns on the same index. The dependency analysis fails to distinguish between different fields (data vs valid) of the same packed struct array element, creating a false combinational loop that the tool cannot resolve.

---

## 2. GitHub Issue 搜索结果

### 搜索关键词
- `arcilator timeout`
- `combinational loop always_comb`
- `dynamic array packed struct`
- `ConvertToArcs`
- `infinite loop`
- `unpacked array`

### 搜索结果概览
- **总搜索关键词数**: 6
- **匹配的 Issue 总数**: 26
- **最高相似度分数**: 4.0/15
- **最相关 Issue**: #9469

---

## 3. 最相关的 Issue 分析

### Issue #9469: [circt-verilog][arcilator] Inconsistent compilation behavior: direct array indexing in always_ff sensitivity list vs. intermediate wire
- **状态**: CLOSED
- **创建者**: m2kar
- **创建时间**: 2026-01-18
- **URL**: https://github.com/llvm/circt/issues/9469
- **相似度分数**: 4.0/15

#### 匹配的关键词
`ConvertToArcs`, `array indexing`

#### Issue 内容摘要
```
## [circt-verilog][arcilator] Inconsistent compilation behavior: direct array indexing in always_ff sensitivity list vs. intermediate wire

## Summary
I encountered an inconsistent compilation behavior in the arcilator flow when using synchronous reset with SystemVerilog array indexing. When directly using array indices (`clkin_data[0]` and `clkin_data[32]`) in an `always_ff` block, the compilation fails with an `llhd.constant_time` error. However, when the same signals are first assigned to int...
```

---

## 4. 其他相关 Issue 排名 (Top 10)

| 排名 | Issue # | 标题 | 分数 | 匹配关键词 | 状态 |
|------|---------|------|------|----------|------|
| 1 | #9469 | [circt-verilog][arcilator] Inconsistent compilatio... | 4.0 | `ConvertToArcs`, `array indexing` | closed |
| 2 | #9560 | [FIRRTL] Canonicalize infinite loop... | 3.0 | `infinite loop` | open |
| 3 | #8022 | [Comb] Infinite loop in OrOp folder... | 3.0 | `infinite loop` | open |
| 4 | #4269 | PrettifyVerilog: infinite loop + growing memory co... | 3.0 | `infinite loop` | open |
| 5 | #8024 | [Comb] Crash in AndOp folder... | 3.0 | `infinite loop` | open |
| 6 | #9420 | [HoistSignals] Multiple drives to same signal are ... | 3.0 | `infinite loop` | closed |
| 7 | #9415 | [FIRRTL] Bytecode reader stucks with debug build... | 3.0 | `infinite loop` | closed |
| 8 | #8865 | [Comb] AddOp canonicalizer hangs in an infinite lo... | 3.0 | `infinite loop` | closed |
| 9 | #5462 | [FIRRTL][CheckCombCycles] Missed cycles, sensitive... | 3.0 | `infinite loop` | closed |
| 10 | #9467 | [circt-verilog][arcilator] `arcilator` fails to lo... | 2.0 | `ConvertToArcs` | open |

---

## 5. 推荐与分析

### 最终建议
**NEW_ISSUE**

### 原因分析
Low similarity score 4.0 - likely unique issue

### 详细分析

#### 相似度评分细节
- **arcilator timeout** (5分): ❌ 未匹配
- **combinational loop** (3分): ❌ 未匹配
- **dynamic array** (2分): ❌ 未匹配
- **packed struct** (2分): ❌ 未匹配
- **always_comb** (3分): ❌ 未匹配
- **ConvertToArcs** (2分): ✓ 匹配 (实际分数: 4.0)
- **array indexing** (2分): ✓ 匹配

#### 差异分析

**本次崩溃的关键特征**:
1. 动态索引数组元素的写-读模式
2. packed struct 字段级别的依赖追踪缺陷
3. 虚假组合循环检测
4. 超时现象（300秒）

**Issue #9469 的特征**:
1. 直接数组索引 vs 中间 wire
2. 敏感度列表中的数组索引
3. 编译行为不一致
4. 无超时（assertion failure）

**结论**: 虽然都涉及数组索引，但 Issue #9469 关注的是敏感度列表中的 always_ff 行为，而本次崩溃关注的是 always_comb 中的动态索引和 packed struct 字段依赖。这是两个不同的问题。

---

## 6. 建议的后续步骤

1. **新建 Issue** 建议
   - 标题: `[circt-verilog][arcilator] Timeout with dynamic-indexed packed struct array in always_comb`
   - 标签: `[Arc]`, `[arcilator]`, `timeout`, `bug`
   - 优先级: 中等 (timeout 问题严重性)

2. **关键信息包含**
   - 最小化的测试用例（source.sv）
   - 完整的错误日志和超时痕迹
   - 涉及的组件和优化通道
   - 根本原因分析

3. **相关组件维护者** (可选抄送)
   - Arc 和 Arcilator 维护者
   - Moore/Verilog 前端维护者

---

## 7. 搜索过程日志

### 执行的搜索查询
- `arcilator timeout`: 找到 0 个相关 Issue
- `combinational loop always_comb`: 找到 0 个相关 Issue
- `dynamic array packed struct`: 找到 0 个相关 Issue
- `ConvertToArcs`: 找到 8 个相关 Issue
- `infinite loop`: 找到 8 个相关 Issue
- `unpacked array`: 找到 10 个相关 Issue

### 搜索参数
- 仓库: `llvm/circt`
- 查询范围: 所有状态 (open/closed)
- 最大结果数: 10 per query
- 评分算法: 关键词权重求和

---

## 附录

### A. 权重映射表

| 关键词 | 权重 |
|---------|------|
| `arcilator timeout` | 5 |
| `combinational loop` | 3 |
| `always_comb` | 3 |
| `infinite loop` | 3 |
| `dynamic array` | 2 |
| `packed struct` | 2 |
| `ConvertToArcs` | 2 |
| `LowerState` | 2 |
| `SplitLoops` | 2 |
| `array indexing` | 2 |
| `unpacked array` | 1 |

### B. 评分规则
- **高相似度** (≥10): 强烈建议审查现有 Issue，可能重复
- **中等相似度** (6-9): 仔细审查现有 Issue，大概率相关但不完全重复
- **低相似度** (<6): 很可能是新的 Issue，但要检查标题和内容

---

**End of Report**
