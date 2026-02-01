# CIRCT Crash Root Cause Analysis

## 1. 崩溃概述

**Testcase ID**: 260128-00000782  
**Crash Type**: Legalization Failure  
**Failed Operation**: `sim.fmt.literal`  
**Pipeline**: circt-verilog → arcilator → opt → llc

## 2. 错误信息

```
<stdin>:4:10: error: failed to legalize operation 'sim.fmt.literal'
    %0 = sim.fmt.literal "Error: Assertion failed: arr["
         ^
<stdin>:4:10: note: see current operation: %28 = "sim.fmt.literal"() <{literal = "Error: Assertion failed: arr["}> : () -> !sim.fstring
```

## 3. 触发代码分析

### 3.1 源码 (source.sv)

```systemverilog
module test_module(
  input logic clk,
  output logic [7:0] arr
);

  logic [2:0] idx;

  always_ff @(posedge clk) begin
    idx <= idx + 1;
  end

  always_comb begin
    arr = 8'b0;
    arr[idx] = 1'b1;
    
    // 触发崩溃的断言
    assert (arr[idx] == 1'b1) else $error("Assertion failed: arr[%0d] != 1", idx);
  end

endmodule
```

### 3.2 关键构造

1. **立即断言 (Immediate Assertion)**: `assert ... else $error(...)`
2. **格式化字符串**: `"Assertion failed: arr[%0d] != 1"` 包含 `%0d` 格式说明符
3. **动态数组索引**: `arr[idx]` 使用变量作为索引

## 4. 根因分析

### 4.1 问题定位

崩溃发生在 **arcilator** 工具的 `LowerArcToLLVM` pass 中。arcilator 是 CIRCT 的硬件仿真后端，负责将 HW/Arc IR 降低到 LLVM IR。

### 4.2 技术根因

1. **断言转换流程**:
   - `circt-verilog` 将 SystemVerilog 立即断言转换为 Moore IR
   - Moore 方言的 `FormatLiteralOp` 转换为 `sim::FormatLiteralOp`
   - 断言的 `$error` 消息被转换为 `sim.fmt.literal` 操作

2. **Legalization 失败原因**:
   - 在 `LowerArcToLLVM.cpp` 中，`sim::FormatLiteralOp` 被标记为 **legal** (line 1087)
   - 这些格式化操作应该由 `SimPrintFormattedProcOpLowering` 模式消费
   - 但如果 `sim.fmt.literal` 在转换过程中被孤立（没有被 `PrintFormattedProcOp` 使用），它将作为"dead code"保留
   - 最终在 legalization 阶段失败，因为没有适当的转换模式处理孤立的 `sim.fmt.literal`

3. **具体问题**:
   - 立即断言 (`assert ... else $error`) 在 `always_comb` 块中
   - arcilator 流程中可能缺少对这种非时钟触发断言的正确处理
   - `sim.fmt.literal` 操作未被正确关联到 `PrintFormattedOp` 或 `PrintFormattedProcOp`

### 4.3 相关源码位置

| 文件 | 位置 | 说明 |
|------|------|------|
| `lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp` | L1087 | 标记 sim.fmt.* 为 legal |
| `lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp` | L809-822 | FormatLiteralOp 处理逻辑 |
| `lib/Dialect/Sim/Transforms/ProceduralizeSim.cpp` | L112 | 检查非 FormatLiteralOp |
| `lib/Conversion/MooreToCore/MooreToCore.cpp` | L1998-2005 | FormatLiteralOp 转换 |

### 4.4 代码路径分析

```
circt-verilog (ImportVerilog)
    ↓ $error("...") 
moore::FormatLiteralOp
    ↓ MooreToCore pass
sim::FormatLiteralOp
    ↓ arcilator pipeline
[缺失] 断言消息未正确绑定到 PrintFormattedOp
    ↓ LowerArcToLLVM
❌ 孤立的 sim.fmt.literal 导致 legalization 失败
```

## 5. 根因结论

**主因**: arcilator 流程对 `always_comb` 块中的立即断言 (`assert ... else $error`) 支持不完整。断言的错误消息格式化操作 (`sim.fmt.literal`) 在转换到 LLVM IR 时成为孤立操作，没有被任何消费者使用，最终触发 legalization 失败。

**分类**: 这是一个 **功能缺失/不完整支持** 类型的 bug，而非传统的崩溃或断言失败。arcilator 工具链对 SystemVerilog 立即断言的仿真支持存在缺口。

## 6. 可能的修复方向

1. 在 arcilator 预处理阶段正确处理立即断言
2. 确保 `sim.fmt.literal` 操作被正确连接到 `PrintFormattedProcOp`
3. 或在 legalization 失败前清理未使用的格式化操作
4. 添加对 `always_comb` 中断言的显式支持或拒绝

## 7. 复现命令

```bash
circt-verilog --ir-hw source.sv | arcilator | opt -O0 | llc -O0 --filetype=obj -o output.o
```
