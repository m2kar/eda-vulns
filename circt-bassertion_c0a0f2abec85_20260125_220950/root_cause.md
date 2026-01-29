# Root Cause Analysis Report

## Executive Summary

`arcilator` 崩溃是因为在 `LowerState` pass 中尝试为 `inout` 端口创建 `arc::StateType`，但该端口的类型是 `!llhd.ref<i8>`（由 Moore → Core 转换产生），而 `StateType` 不支持 LLHD 的引用类型。Arc 方言明确不支持 `inout` 端口，但缺少早期的验证检查，导致在类型创建时触发断言失败而非给出用户友好的错误消息。

## Crash Context

- **Tool**: arcilator
- **Dialect**: Arc (with LLHD types from Moore conversion)
- **Failing Pass**: LowerStatePass
- **Crash Type**: Assertion failure
- **CIRCT Version**: 1.139.0

## Error Analysis

### Assertion Message
```
Assertion `succeeded(ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Error Message
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i8>'
```

### Key Stack Frames
```
#12 circt::arc::StateType::get(mlir::Type) ArcTypes.cpp.inc:108
#13 ModuleLowering::run() LowerState.cpp:219
#15 LowerStatePass::runOnOperation() LowerState.cpp:1198
```

## Test Case Analysis

### Code Summary
```systemverilog
module MixedPorts(
  input  logic       clk,
  input  logic       a,
  output logic       b,
  inout  logic [7:0] c,      // <-- 问题端口
  output reg   [7:0] count
);
  logic drive_enable;
  assign c = drive_enable ? count : 8'bz;  // Tri-state driver
  always_ff @(posedge clk) begin count <= count + 1; end
  always_comb begin drive_enable = a; b = a; end
endmodule
```

### Key Constructs
- `inout logic [7:0] c` - 双向端口声明
- `8'bz` - 高阻态赋值（tri-state）
- Tri-state driver 条件赋值

### Problematic Pattern
SystemVerilog 的 `inout` 端口在 `circt-verilog --ir-hw` 管道中被转换为 Moore 方言的 `RefType`，然后通过 `MooreToCore` 转换为 `!llhd.ref<i8>`。当 IR 进入 `arcilator` 时，`LowerState` pass 尝试为所有模块参数创建 `arc::StateType`，但 `llhd::RefType` 不是 `StateType` 所支持的类型。

## CIRCT Source Analysis

### Crash Location
**File**: `lib/Dialect/Arc/Transforms/LowerState.cpp:219`
**Function**: `ModuleLowering::run()`

```cpp
// Allocate storage for the inputs.
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(arg.getType()), name, storageArg);
  //                      ^^^^^^^^^^^^^^^^^^^^^^^^
  //                      CRASH HERE: arg.getType() = !llhd.ref<i8>
  allocatedInputs.push_back(state);
}
```

### Type Verification
**File**: `lib/Dialect/Arc/ArcTypes.cpp:80-87`

```cpp
LogicalResult StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                                Type innerType) {
  if (!computeLLVMBitWidth(innerType))
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

### Bit Width Computation
**File**: `lib/Dialect/Arc/ArcTypes.cpp:29-76`

`computeLLVMBitWidth()` 函数只支持以下类型：
- `seq::ClockType`
- `IntegerType`
- `hw::ArrayType`
- `hw::StructType`

**不支持**: `llhd::RefType`（LLHD 的引用类型）

### ModelOp Verification (Late Check)
**File**: `lib/Dialect/Arc/ArcOps.cpp:337-341`

```cpp
for (const hw::ModulePort &port : getIo().getPorts())
  if (port.dir == hw::ModulePort::Direction::InOut)
    return emitOpError("inout ports are not supported");
```

这个检查存在，但只在 `ModelOp` 验证时执行，而 `LowerState` pass 在模块转换 **之前** 就尝试处理端口参数。

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: `LowerState` pass 缺少对 `inout` 端口（`!llhd.ref` 类型）的早期检查

**Evidence**:
1. 错误消息明确指出 `!llhd.ref<i8>` 类型没有已知的位宽
2. `computeLLVMBitWidth()` 不处理 `llhd::RefType`，返回空 optional
3. `StateType::verify()` 对不支持的类型返回失败，但这在 `StateType::get()` 内部触发断言
4. `ModelOp::verify()` 有 `inout` 端口检查，但执行顺序太晚

**Mechanism**:
```
circt-verilog --ir-hw input.sv
  → Moore dialect (inout → RefType)
  → MooreToCore (RefType → !llhd.ref<i8>)
  
arcilator receives IR
  → Preprocessing passes (StripSV, etc.)
  → LowerStatePass::run()
    → ModuleLowering::run()
      → For each module argument:
          → StateType::get(arg.getType())  // arg.getType() = !llhd.ref<i8>
            → StateType::verify() → FAILS
              → ASSERTION FAILURE
```

### Hypothesis 2 (Medium Confidence)
**Cause**: `arcilator` 的预处理管道缺少 `HWEliminateInOutPorts` pass

**Evidence**:
1. CIRCT 有 `HWEliminateInOutPorts` pass 可以将 inout 端口转换为分离的 input/output
2. `populateArcPreprocessingPipeline()` 不包含此 pass
3. 如果添加此 pass，inout 端口会在 `LowerState` 之前被消除

**Limitation**:
- 该 pass 可能无法处理所有 inout 使用场景（如 tri-state 驱动）
- 该 pass 的注释提到 "hw.inout outputs not yet supported"

### Hypothesis 3 (Low Confidence)
**Cause**: `computeLLVMBitWidth()` 应该扩展以支持 `llhd::RefType`

**Evidence**:
1. `llhd::RefType` 本质上是指向另一个类型的引用
2. 可以通过获取内部类型来计算位宽

**Counter-evidence**:
- LLHD 引用类型语义上与 Arc 的状态模型不兼容
- 正确的做法是拒绝 inout 端口，而非尝试支持

## Suggested Fix Directions

### Fix 1 (Recommended): 在 `LowerState` 开始时添加 inout 端口检查

```cpp
// In ModuleLowering::run(), before processing arguments:
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  if (isa<llhd::RefType>(arg.getType())) {
    return moduleOp.emitOpError()
        << "inout ports are not supported in arcilator; "
        << "argument " << arg.getArgNumber() << " has type " << arg.getType();
  }
}
```

### Fix 2: 在 `arcilator` 入口点添加检查

在 `populateArcPreprocessingPipeline()` 之前添加一个验证 pass，检查模块是否包含 inout 端口。

### Fix 3: 改进错误消息

当 `StateType::verify()` 失败时，提供更具体的诊断信息，指出 arcilator 不支持 inout 端口。

## Keywords for Issue Search

`inout` `llhd.ref` `StateType` `LowerState` `arcilator` `bit width` `RefType` `tri-state`

## Related Files

- `lib/Dialect/Arc/Transforms/LowerState.cpp` - 崩溃位置
- `lib/Dialect/Arc/ArcTypes.cpp` - 类型验证逻辑
- `lib/Dialect/Arc/ArcOps.cpp` - ModelOp 验证（有 inout 检查但执行太晚）
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - RefType 转换为 llhd::RefType
- `lib/Dialect/SV/Transforms/HWEliminateInOutPorts.cpp` - 可能的解决方案参考

## Conclusion

这是一个 **确认的 Bug**：`arcilator` 明确不支持 `inout` 端口（有验证代码证明），但缺少早期检查导致在类型创建时触发断言失败。正确的行为应该是在遇到 `inout` 端口时给出清晰的用户级错误消息，而非崩溃。
