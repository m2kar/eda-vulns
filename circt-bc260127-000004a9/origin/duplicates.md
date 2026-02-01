# CIRCT Bug 重复检查报告

**报告生成时间**: 2026-01-31 18:35:53

## 原始测试用例

- **测试用例 ID**: 260127-000004a9
- **崩溃类型**: assertion
- **工具**: arcilator
- **方言**: arc, llhd, hw
- **错误消息**: `state type must have a known bit width; got '!llhd.ref<i1>'`

### 崩溃位置

- 文件: `lib/Dialect/Arc/Transforms/LowerState.cpp`
- 行号: 219
- 函数: `ModuleLowering::run()`

### 根本原因

- **分类**: unsupported_type
- **描述**: LowerState pass attempts to create arc::StateType from llhd::RefType (inout port), but StateType::verify() fails because computeLLVMBitWidth() cannot compute bit width for RefType
- **触发构造**: inout port in SystemVerilog
- **IR 类型**: `!llhd.ref<i1>`

---

## 搜索结果概览

- **搜索查询数**: 5
- **发现不同 Issues**: 36
- **最相似 Issue**: #9467
- **最高相似度**: 42/100

---

## 重复检查建议

### **建议**: `LIKELY_NEW`

**理由**: 找到部分相关 Issue（分数: 42/100），但不够高，可能是新问题

---

## 详细搜索结果（前 15 名）


### 1. Issue #9467: [circt-verilog][arcilator] `arcilator` fails to lower `llhd.constant_time` generated from simple SV delay (`#1`)

- **相似度分数**: 42/100
- **状态**: OPEN
- **最后更新**: 2026-01-20
- **链接**: [https://github.com/llvm/circt/issues/9467](https://github.com/llvm/circt/issues/9467)
- **关键匹配**: 匹配(3): arc,arcilator,lowering, 标签(2)
- **内容预览**: 
  ```
  ## Title
[circt-verilog][arcilator] `arcilator` fails to lower `llhd.constant_time` generated from simple SV delay (`#1`)

## Summary
I ran into a failure in the arcilator flow with a minimal SystemVe...
  ```


### 2. Issue #7703: [Arc] Improve LowerState to never produce read-after-write conflicts

- **相似度分数**: 40/100
- **状态**: CLOSED
- **最后更新**: 2024-10-28
- **链接**: [https://github.com/llvm/circt/pull/7703](https://github.com/llvm/circt/pull/7703)
- **关键匹配**: 匹配(3): lowerstate,arc,lowering, 标签(1)
- **内容预览**: 
  ```
  This is a complete rewrite of the `LowerState` pass that makes the `LegalizeStateUpdate` pass obsolete.

The old implementation of `LowerState` produces `arc.model`s that still contain read-after-wr...
  ```


### 3. Issue #9395: [circt-verilog][arcilator] Arcilator assertion failure

- **相似度分数**: 37/100
- **状态**: CLOSED
- **最后更新**: 2026-01-19
- **链接**: [https://github.com/llvm/circt/issues/9395](https://github.com/llvm/circt/issues/9395)
- **关键匹配**: 匹配(3): assertion,arc,arcilator, 标签(1)
- **内容预览**: 
  ```
  Hi, all! Let's look at this example on _Verilog_:

```
module comb_assert(
    input wire clk,
    input wire resetn
);
    always @* begin
        if (resetn) begin
            assert (0);
        en...
  ```


### 4. Issue #8825: [LLHD] Switch from hw.inout to a custom signal reference type

- **相似度分数**: 35/100
- **状态**: OPEN
- **最后更新**: 2025-08-06
- **链接**: [https://github.com/llvm/circt/issues/8825](https://github.com/llvm/circt/issues/8825)
- **关键匹配**: 匹配(2): inout,llhd.ref, 标签(1)
- **内容预览**: 
  ```
  A while ago, we have switched the LLHD dialect from a custom `!llhd.sig` wrapper type to `!hw.inout` in order to represent a reference to a signal slot that can be probed and driven. To support Verilo...
  ```


### 5. Issue #8012: [Moore][Arc][LLHD] Moore to LLVM lowering issues

- **相似度分数**: 34/100
- **状态**: OPEN
- **最后更新**: 2024-12-22
- **链接**: [https://github.com/llvm/circt/issues/8012](https://github.com/llvm/circt/issues/8012)
- **关键匹配**: 匹配(3): arc,arcilator,lowering
- **内容预览**: 
  ```
  Hi all!

I am trying to simulate a SystemVerilog code (listed below) using `arcilator`:

```verilog
module dff(D, clk, Q);
    input D; // Data input 
    input clk; // clock input 
    output reg Q; ...
  ```


### 6. Issue #8065: [LLHD][Arc] Indexing and slicing lowering from Verilog to LLVM IR

- **相似度分数**: 34/100
- **状态**: OPEN
- **最后更新**: 2025-02-11
- **链接**: [https://github.com/llvm/circt/issues/8065](https://github.com/llvm/circt/issues/8065)
- **关键匹配**: 匹配(3): arc,arcilator,lowering
- **内容预览**: 
  ```
  Hi! I'm trying to use circt for lowering to LLVM IR. I found such a construction in some example:

```verilog
module Mod (input clk, input a, input b, output logic[1:0] c);
always_ff @(posedge clk) be...
  ```


### 7. Issue #7676: [RFC][Sim] Add triggered simulation procedures

- **相似度分数**: 34/100
- **状态**: OPEN
- **最后更新**: 2025-06-02
- **链接**: [https://github.com/llvm/circt/pull/7676](https://github.com/llvm/circt/pull/7676)
- **关键匹配**: 匹配(3): arc,arcilator,lowering
- **内容预览**: 
  ```
  Continuing the series of #7314 and #7335 (and hoping to finally get to lower the `sim.proc.print` operation) this PR adds trigger-related types and operations to the Sim Dialect. The primary point is ...
  ```


### 8. Issue #6810: [Arc] Add basic assertion support

- **相似度分数**: 33/100
- **状态**: OPEN
- **最后更新**: 2024-03-15
- **链接**: [https://github.com/llvm/circt/issues/6810](https://github.com/llvm/circt/issues/6810)
- **关键匹配**: 匹配(3): assertion,arc,lowering, 标签(1)
- **内容预览**: 
  ```
  Add support for `verif.assert` and `sv.assert.concurrent` operations to the Arc dialect and passes. When lowering towards LLVM, the asserts should lower to an `scf.if` operation that checks whether th...
  ```


### 9. Issue #9466: [circt-verilog][arcilator] `arcilator` fails to lower `llhd.constant_time` generated from simple SV delay (`#1`)

- **相似度分数**: 31/100
- **状态**: CLOSED
- **最后更新**: 2026-01-17
- **链接**: [https://github.com/llvm/circt/issues/9466](https://github.com/llvm/circt/issues/9466)
- **关键匹配**: 匹配(3): arc,arcilator,lowering
- **内容预览**: 
  ```
  ## Summary
I ran into a failure in the arcilator flow with a minimal SystemVerilog module using a single `#1` delay. The same design runs under Verilator, but arcilator reports `llhd.constant_time` as...
  ```


### 10. Issue #9469: [circt-verilog][arcilator] Inconsistent compilation behavior: direct array indexing in always_ff sensitivity list vs. intermediate wire

- **相似度分数**: 31/100
- **状态**: CLOSED
- **最后更新**: 2026-01-25
- **链接**: [https://github.com/llvm/circt/issues/9469](https://github.com/llvm/circt/issues/9469)
- **关键匹配**: 匹配(2): arc,arcilator, 标签(2)
- **内容预览**: 
  ```
  ## [circt-verilog][arcilator] Inconsistent compilation behavior: direct array indexing in always_ff sensitivity list vs. intermediate wire

## Summary
I encountered an inconsistent compilation behavio...
  ```


### 11. Issue #9442: [Arc] Add nascent support for sim.proc.print and sim.fmt.*

- **相似度分数**: 30/100
- **状态**: CLOSED
- **最后更新**: 2026-01-20
- **链接**: [https://github.com/llvm/circt/pull/9442](https://github.com/llvm/circt/pull/9442)
- **关键匹配**: 匹配(3): assertion,arc,lowering, 标签(1)
- **内容预览**: 
  ```
  Supports sim.fmt.* during ArcToLLVM lowering by converting the format ops
into a printf format string, then calling printf during the lowering of
`sim.proc.print`.

Ideally this would eventually be a ...
  ```


### 12. Issue #6378: [HW to BTOR2] btor2 conversion pass

- **相似度分数**: 26/100
- **状态**: CLOSED
- **最后更新**: 2023-12-15
- **链接**: [https://github.com/llvm/circt/pull/6378](https://github.com/llvm/circt/pull/6378)
- **关键匹配**: 匹配(3): assertion,arc,lowering
- **内容预览**: 
  ```
  ## TLDR  
This PR introduces a btor emission pass that converts flattened (as in with inlined sub-modules) designs from the hw dialect into the `btor2` format used for bounded model checking. This PR...
  ```


### 13. Issue #6783: [arcilator] Introduce integrated JIT for simulation execution

- **相似度分数**: 23/100
- **状态**: CLOSED
- **最后更新**: 2024-03-20
- **链接**: [https://github.com/llvm/circt/pull/6783](https://github.com/llvm/circt/pull/6783)
- **关键匹配**: 匹配(2): arc,arcilator
- **内容预览**: 
  ```
  This PR adds a JIT runtime for arcilator, backed by MLIR's ExecutionEngine. This JIT allows executing `arc.sim` operations directly from the arcilator binary....
  ```


### 14. Issue #8870: [LLHD] Remove redundant destination operands from wait operation

- **相似度分数**: 23/100
- **状态**: OPEN
- **最后更新**: 2025-09-04
- **链接**: [https://github.com/llvm/circt/pull/8870](https://github.com/llvm/circt/pull/8870)
- **关键匹配**: 匹配(2): arc,lowering, 标签(1)
- **内容预览**: 
  ```
  While lowering Rocket Chip from arc-tests through the `fir` -> `verilog` -> `moore` -> `core` -> `llhd` pipeline, a template is generated in which the `clock` module input is propagated through all co...
  ```


### 15. Issue #8950: [ConvertToArcs] Add llhd.combinational conversion

- **相似度分数**: 20/100
- **状态**: CLOSED
- **最后更新**: 2025-09-18
- **链接**: [https://github.com/llvm/circt/pull/8950](https://github.com/llvm/circt/pull/8950)
- **关键匹配**: 匹配(2): arc,lowering, 标签(1)
- **内容预览**: 
  ```
  Introduce a dialect conversion step into the `ConvertToArcs` pass. We'll use this to map various core dialect operations to Arc-specific ones in the future. As a starting point, add a conversion from ...
  ```


---

## 分析和建议

### 匹配特征

本测试用例的关键特征：
1. **工具**: arcilator - CIRCT 的高级综合模拟器
2. **方言**: Arc, LLHD, HW - 主要关注状态降低和仿真
3. **错误**: StateType 无法验证 RefType（inout 端口）
4. **核心问题**: LowerState pass 在处理 inout 端口时缺乏类型检查

### 相似 Issue 分析

**高分 Issue (#9467, #7703)**:
- 都涉及 arcilator 和 LowerState
- 处理降低过程中的边界情况
- 可能共享相同的根本原因

**中分 Issue (#8825)**:
- 直接处理 inout 和 llhd.ref
- 可能提供解决方案参考

**底分 Issue**:
- 涉及 Arc/LLHD 但没有直接的 inout/StateType 关联
- 可能的背景上下文

### 建议的后续步骤

1. 检查 #9467 和 #7703 的内容，确认是否存在重复
2. 如果不重复，继续以新 Issue 报告
3. 关键应包含：
   - inout 端口的 SystemVerilog 测试用例
   - StateType 验证的类型检查需求
   - LowerState pass 的处理能力限制
   - 建议的修复方案

---

## 元数据

- **搜索时间**: {datetime.now().isoformat()}
- **搜索引擎**: GitHub API
- **相似度算法**: 关键词匹配 + 标签评分
- **置信度**: {'高' if top_score >= 40 else '中' if top_score >= 20 else '低'}

