# Root Cause Analysis Report

## Executive Summary

arcilator 的 LowerState pass 在处理包含 inout 端口和三态条件赋值的 SystemVerilog 模块时崩溃。根本原因是 `llhd::RefType` 类型无法被 `arc::StateType::get()` 验证通过，因为 `computeLLVMBitWidth()` 函数不支持 LLHD ref 类型，导致位宽计算返回空值。

## Crash Context

- **Tool/Command**: `circt-verilog --ir-hw | arcilator`
- **Dialect**: Arc/LLHD
- **Failing Pass**: LowerStatePass (Arc Transforms)
- **Crash Type**: Assertion failure
- **Crash Location**: `LowerState.cpp:219` -> `StateType::get()`

## Error Analysis

### Assertion/Error Message

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: .../mlir/include/mlir/IR/StorageUniquerSupport.h:180: 
Assertion `succeeded(ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Key Stack Frames

```
#12 circt::arc::StateType::get(mlir::Type) ArcTypes.cpp.inc:108
#13 (anonymous namespace)::ModuleLowering::run() LowerState.cpp:219
#14 (anonymous namespace)::LowerStatePass::runOnOperation() LowerState.cpp:1198
```

## Test Case Analysis

### Code Summary

```systemverilog
module my_module (
  input  logic enable,
  inout  logic io_sig
);
  logic out_val;
  assign out_val = enable;
  assign io_sig = (out_val) ? 1'b1 : 1'bz;
endmodule
```

该模块包含一个 **inout 双向端口** `io_sig`，并使用三态条件赋值：当 `out_val` 为真时输出 `1'b1`，否则输出高阻态 `1'bz`。

### Key Constructs

| 构造 | 与崩溃的关系 |
|------|-------------|
| `inout logic io_sig` | 双向端口被转换为 `llhd::RefType`，用于表示可读写的信号引用 |
| `assign io_sig = ... ? 1'b1 : 1'bz` | 三态条件赋值需要对 inout 端口进行写操作 |
| `1'bz` 高阻态值 | 三态逻辑是导致使用 LLHD 方言的原因 |

### Potentially Problematic Patterns

1. **inout 端口 + 三态赋值组合**：circt-verilog 使用 LLHD 方言处理双向端口和三态逻辑
2. **LLHD -> Arc 方言转换缺口**：arcilator 的 LowerState pass 无法处理 `llhd::RefType`

## CIRCT Source Analysis

### Crash Location

**File**: `lib/Dialect/Arc/ArcTypes.cpp`  
**Function**: `StateType::verify()`  
**Line**: ~87-91

### Code Context

```cpp
// lib/Dialect/Arc/ArcTypes.cpp
static std::optional<uint64_t> computeLLVMBitWidth(Type type) {
  if (isa<seq::ClockType>(type))
    return 1;

  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();

  if (auto arrayType = dyn_cast<hw::ArrayType>(type)) { ... }
  if (auto structType = dyn_cast<hw::StructType>(type)) { ... }

  // We don't know anything about any other types.
  return {};  // <-- llhd::RefType 会走到这里，返回空
}

LogicalResult StateType::verify(..., Type innerType) {
  if (!computeLLVMBitWidth(innerType))  // <-- llhd::RefType 验证失败
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

在 `LowerState.cpp:219` 的调用链：

```cpp
// lib/Dialect/Arc/Transforms/LowerState.cpp:219
// ModuleLowering::getAllocatedState() 中
auto alloc = AllocStateOp::create(allocBuilder, result.getLoc(),
                                  StateType::get(result.getType()),  // <-- 崩溃点
                                  storageArg);
```

当 `result.getType()` 是 `llhd::RefType<i1>` 时，`StateType::get()` 会触发验证失败。

### Processing Path

1. **circt-verilog --ir-hw** 将 SystemVerilog 编译为 HW/LLHD 混合 IR
2. **inout 端口** 被表示为 `llhd::RefType<i1>` 类型
3. **arcilator** 运行 LowerState pass 试图为所有模块结果分配存储
4. **ModuleLowering::getAllocatedState()** 调用 `StateType::get(result.getType())`
5. `result.getType()` 是 `llhd::RefType<i1>`
6. **StateType::verify()** 调用 `computeLLVMBitWidth(llhd::RefType<i1>)`
7. `computeLLVMBitWidth()` **不识别 llhd::RefType**，返回 `std::nullopt`
8. 验证失败，触发 assertion

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence): arcilator 不支持 LLHD RefType

**Cause**: `arc::StateType` 的验证逻辑 (`computeLLVMBitWidth`) 只支持有限的类型集合（`seq::ClockType`、`IntegerType`、`hw::ArrayType`、`hw::StructType`），不包括 `llhd::RefType`。

**Evidence**:
- 错误信息明确指出 "state type must have a known bit width; got '!llhd.ref<i1>'"
- `computeLLVMBitWidth()` 源码显示只处理 4 种类型
- `llhd::RefType` 不在支持的类型列表中

**Mechanism**: 
当 inout 端口产生的 `llhd::RefType` 值传递给 `StateType::get()` 时，验证失败导致 assertion。

### Hypothesis 2 (Medium Confidence): circt-verilog 生成的 IR 不适合 arcilator

**Cause**: circt-verilog 对 inout/tristate 使用 LLHD 方言，但 arcilator 期望纯 HW/Seq/Comb IR。

**Evidence**:
- 管道流程：`circt-verilog --ir-hw | arcilator`
- 用户期望 `--ir-hw` 产生 HW 方言 IR
- 实际产生了混合的 HW + LLHD IR

**Mechanism**:
arcilator 设计用于模拟 HW 层级的设计，不处理 LLHD 的信号语义（包括 ref 类型）。

### Hypothesis 3 (Low Confidence): 缺少必要的转换 Pass

**Cause**: 在 arcilator 之前可能需要一个 LLHD -> HW 的转换 pass 来处理 RefType。

**Evidence**:
- LLHD 方言有自己的仿真器 (llhd-sim)
- arcilator 专门处理 HW/Arc 方言
- 可能缺少跨方言转换

## Suggested Fix Directions

1. **在 arcilator 中添加 llhd::RefType 支持**
   - 修改 `computeLLVMBitWidth()` 以支持 `RefType`
   - 可能需要解包 ref 内部类型：`refType.getNestedType()`
   
2. **添加验证/诊断**
   - 在 LowerStatePass 开始时检测不支持的 LLHD 类型
   - 产生更友好的错误信息而非 assertion 崩溃

3. **文档/工具链改进**
   - 文档说明 arcilator 不支持 inout/tristate 逻辑
   - 或添加警告：当检测到 LLHD 方言时提示用户使用 llhd-sim

4. **circt-verilog 改进**
   - 提供选项禁用/转换 LLHD 特性
   - 或添加 pass 将 RefType 转换为其他表示

## Keywords for Issue Search

`arcilator` `llhd.ref` `StateType` `bit width` `inout` `tristate` `LowerState` `computeLLVMBitWidth`

## Related Files to Investigate

| 文件 | 原因 |
|------|------|
| `lib/Dialect/Arc/ArcTypes.cpp` | 包含 `computeLLVMBitWidth()` 和 `StateType::verify()` |
| `lib/Dialect/Arc/Transforms/LowerState.cpp` | 崩溃位置，需要添加类型检查 |
| `include/circt/Dialect/Arc/ArcTypes.td` | StateType 定义 |
| `lib/Conversion/MooreToCore/MooreToCore.cpp` | inout 端口如何被转换为 llhd::RefType |

## Additional Notes

- **影响范围**: 任何使用 inout 端口或三态逻辑的 SystemVerilog 设计通过 arcilator 仿真都会触发此崩溃
- **严重程度**: High - 完全阻止仿真这类设计
- **临时解决方案**: 使用 llhd-sim 而非 arcilator，或重构设计避免 inout/tristate
