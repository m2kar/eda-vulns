# Root Cause Analysis Report

## Executive Summary

`arcilator` 在处理包含 **inout 端口**的模块时崩溃。测例声明了 `inout logic c`，经过 `circt-verilog --ir-hw` 转换后产生 `!llhd.ref<i1>` 类型。当 `LowerState` pass 尝试为该端口创建 `arc::StateType` 时，验证失败触发断言，因为 `!llhd.ref` 不是 Arc dialect 支持的类型。

## Crash Context

- **Tool/Command**: `arcilator`
- **Dialect**: Arc
- **Failing Pass**: `LowerState` (`arc-lower-state`)
- **Crash Type**: Assertion failure
- **CIRCT Version**: 1.139.0

## Error Analysis

### Error Message
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
```

### Assertion Message
```cpp
Assertion `succeeded(ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Key Stack Frames
```
#11 circt::arc::StateType::get(mlir::Type)
    ArcTypes.cpp.inc:108
#12 (anonymous namespace)::ModuleLowering::run()
    LowerState.cpp:219
#14 (anonymous namespace)::LowerStatePass::runOnOperation()
    LowerState.cpp:1198
```

## Test Case Analysis

### Code Summary
```systemverilog
module MixedPorts(
  input  logic clk,
  input  logic rst,
  input  logic a,
  output logic b,
  inout  logic c    // <-- 问题所在
);
```

这是一个简单的带有混合端口方向的模块，包括一个 **inout（双向）端口** `c`。

### Key Constructs
- `inout logic c`: 双向端口，这是触发崩溃的关键构造
- `always_ff`: 时序逻辑块
- 三态赋值: `assign c = (!rst) ? 1'bz : a;`

### Problematic Pattern
**SystemVerilog inout 端口** → 被转换为 `!llhd.ref<i1>` 类型 → Arc dialect 不支持

## CIRCT Source Analysis

### Crash Location
**File**: `lib/Dialect/Arc/Transforms/LowerState.cpp`
**Function**: `ModuleLowering::run()`
**Line**: 219

### Code Context
```cpp
// lib/Dialect/Arc/Transforms/LowerState.cpp:214-221
// Allocate storage for the inputs.
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(arg.getType()), name, storageArg);
                          // ^^^^^^^^^^^^^^^^^^^^^^^^
                          // arg.getType() = !llhd.ref<i1> for inout port
                          // StateType::get() validation fails
  allocatedInputs.push_back(state);
}
```

### StateType Validation Logic
```cpp
// lib/Dialect/Arc/ArcTypes.cpp:80-86
LogicalResult
StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Type innerType) {
  if (!computeLLVMBitWidth(innerType))
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

`computeLLVMBitWidth()` 只能处理以下类型：
- `seq::ClockType`
- `IntegerType`
- `hw::ArrayType`
- `hw::StructType`

`!llhd.ref<T>` 不在支持列表中。

### Arc ModelOp Verification
```cpp
// lib/Dialect/Arc/ArcOps.cpp:337-339
for (const hw::ModulePort &port : getIo().getPorts())
  if (port.dir == hw::ModulePort::Direction::InOut)
    return emitOpError("inout ports are not supported");
```

Arc dialect **设计上就不支持 inout 端口**，但这个检查是在 `ModelOp::verify()` 中，而崩溃发生在 `LowerState` pass 更早的阶段。

### Processing Path
1. `circt-verilog --ir-hw` 将 SystemVerilog 转换为 HW dialect
2. inout 端口被表示为 `!llhd.ref<i1>` 类型
3. `arcilator` 运行 `LowerState` pass
4. Pass 遍历所有模块参数，尝试创建 `RootInputOp`
5. `StateType::get(arg.getType())` 被调用，其中 `arg.getType() = !llhd.ref<i1>`
6. `StateType::verify()` 调用 `computeLLVMBitWidth(!llhd.ref<i1>)`
7. `computeLLVMBitWidth()` 返回 `std::nullopt`（不认识该类型）
8. **验证失败 → 断言触发 → 崩溃**

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence) ✓
**Cause**: Arc dialect 不支持 inout 端口，但缺少早期检测导致在 LowerState pass 中崩溃

**Evidence**:
1. 测例使用 `inout logic c` 端口
2. 错误消息明确指出 `!llhd.ref<i1>` 类型无法创建 StateType
3. `ArcOps.cpp:338-339` 证实 Arc 设计上不支持 inout
4. Stack trace 确认崩溃点在 `LowerState.cpp:219`

**Mechanism**:
- `circt-verilog --ir-hw` 将 inout 端口转换为 `!llhd.ref<T>` 类型
- `arcilator` 的 pipeline 没有在早期检测到这个不支持的端口类型
- 当 `LowerState` pass 尝试为端口分配状态存储时，`StateType::get()` 验证失败
- 验证失败导致断言触发而不是优雅的错误报告

### Hypothesis 2 (Medium Confidence)
**Cause**: `computeLLVMBitWidth()` 应该处理更多类型或返回有意义的错误

**Evidence**:
- 函数只处理 4 种类型，遇到其他类型返回空
- `llhd.ref` 是 LLHD dialect 的引用类型，理论上可以计算其位宽

**Note**: 这不是根本问题，因为 Arc 本身就不支持 inout 端口

## Suggested Fix Directions

### Option 1: 早期检测不支持的端口类型 (推荐)
在 `arcilator` pipeline 的早期阶段添加检查，当检测到 inout 端口时发出清晰的错误消息：

```cpp
// 在 LowerState pass 开始前或 arcilator 入口处
for (auto port : module.getPorts()) {
  if (port.dir == hw::ModulePort::Direction::InOut) {
    return module.emitError()
           << "inout ports are not supported by arcilator";
  }
}
```

### Option 2: 优雅处理 StateType 创建失败
修改 `LowerState.cpp:219` 处的代码，在 `StateType::get()` 失败时返回有意义的错误而不是断言：

```cpp
auto stateType = StateType::getChecked(
    [&]() { return emitError(arg.getLoc()); },
    arg.getType());
if (!stateType)
  return failure();  // 而不是断言
```

### Option 3: 扩展支持（复杂）
如果 Arc 需要支持 inout 端口，则需要：
1. 扩展 `computeLLVMBitWidth()` 处理 `llhd.ref` 类型
2. 修改 `ModelOp::verify()` 移除 inout 限制
3. 实现 inout 端口的完整语义

## Keywords for Issue Search
`arcilator` `inout` `StateType` `llhd.ref` `LowerState` `known bit width`

## Related Files to Investigate
- `lib/Dialect/Arc/Transforms/LowerState.cpp` - 崩溃点
- `lib/Dialect/Arc/ArcTypes.cpp` - StateType 验证逻辑
- `lib/Dialect/Arc/ArcOps.cpp` - ModelOp inout 检查
- `tools/arcilator/arcilator.cpp` - 工具入口点

## Classification
- **Bug Type**: Missing early validation / Assertion instead of error
- **Severity**: Medium
- **Component**: Arc dialect / arcilator
- **Root Cause**: Unsupported feature (inout ports) not detected early enough
