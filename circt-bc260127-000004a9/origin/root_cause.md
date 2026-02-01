# CIRCT 崩溃根因分析报告

## 崩溃概要

| 项目 | 内容 |
|------|------|
| **崩溃类型** | Assertion Failure |
| **崩溃工具** | arcilator |
| **崩溃位置** | `LowerState.cpp:219` → `StateType::get()` |
| **方言** | Arc, LLHD, HW |
| **严重性** | High |

## 错误信息

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: ...StorageUniquerSupport.h:180: Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

## 问题描述

### 触发场景

测例代码 `source.sv` 包含一个带有 **inout (双向/三态)端口** 的 Verilog 模块:

```systemverilog
module MixPorts(
  input  logic [63:0] wide_input,
  output logic [31:0] out_val,
  inout  logic        io_sig     // <-- 触发问题的 inout 端口
);
  // ...
  assign io_sig = (out_val[0]) ? 1'b1 : 1'bz;  // 三态赋值
endmodule
```

### 根因分析

1. **inout 端口的 LLHD 表示**: 当 SystemVerilog 的 `inout` 端口通过 `circt-verilog --ir-hw` 转换时，在 LLHD 方言中被表示为 `!llhd.ref<i1>` 类型（引用类型）。

2. **arcilator 的 LowerState pass**: 在 `LowerState.cpp:219` 处，代码尝试为模块输入创建 `arc::StateType`:
   ```cpp
   auto state = RootInputOp::create(allocBuilder, arg.getLoc(),
                   StateType::get(arg.getType()), name, storageArg);
   ```

3. **StateType 验证失败**: `arc::StateType` 需要内部类型具有已知的位宽。验证函数 `StateType::verify()` 在 `ArcTypes.cpp:81-86` 中调用 `computeLLVMBitWidth()`:
   ```cpp
   LogicalResult StateType::verify(..., Type innerType) {
     if (!computeLLVMBitWidth(innerType))
       return emitError() << "state type must have a known bit width; got "
                          << innerType;
     return success();
   }
   ```

4. **位宽计算失败**: `computeLLVMBitWidth()` 函数只能处理以下类型:
   - `seq::ClockType`
   - `IntegerType`
   - `hw::ArrayType`
   - `hw::StructType`
   
   **但无法处理 `llhd::RefType`**，因此返回 `std::nullopt`，导致验证失败。

5. **断言触发**: 由于 `verifyInvariants` 失败，MLIR 的 `StorageUniquerSupport.h:180` 触发断言。

### 调用栈关键帧

```
#12 circt::arc::StateType::get(mlir::Type)
#13 (anonymous namespace)::ModuleLowering::run()  [LowerState.cpp:219]
#14 (anonymous namespace)::LowerStatePass::runOnOperation()
```

## 问题根源

**arcilator 的 LowerState pass 未能正确处理 LLHD 的 `RefType`（来自 inout 端口）**。

- 应该在尝试创建 `StateType` 之前检查类型是否有效
- 或者在更早的阶段将 inout 端口转换为 arcilator 支持的表示形式
- 或者发出有意义的诊断而非断言失败

## 受影响的代码路径

1. `circt-verilog --ir-hw`: 将 inout 端口转换为 `!llhd.ref` 类型
2. `arcilator` → `LowerState` pass: 无法处理 `!llhd.ref` 类型

## 复现流程

```bash
circt-verilog --ir-hw source.sv | arcilator
```

## 建议修复方向

1. **选项 1**: 在 `LowerState.cpp` 中添加对 `llhd::RefType` 的检查，遇到不支持的类型时发出有意义的错误诊断
2. **选项 2**: 扩展 `computeLLVMBitWidth()` 以支持 `llhd::RefType`
3. **选项 3**: 在 arcilator 流程早期拒绝/转换包含 inout 端口的模块

## 相关源码文件

- `/lib/Dialect/Arc/Transforms/LowerState.cpp` - 崩溃发生位置
- `/lib/Dialect/Arc/ArcTypes.cpp` - StateType 验证逻辑
- `/lib/Dialect/LLHD/IR/LLHDTypes.td` - RefType 定义
