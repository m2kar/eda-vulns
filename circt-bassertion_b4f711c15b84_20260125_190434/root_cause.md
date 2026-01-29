# Root Cause Analysis Report

## Executive Summary

`arcilator` 的 `LowerStatePass` 在处理包含 `inout` 端口的模块时崩溃。当尝试为模块输入分配存储时，`StateType::get()` 被调用来包装 inout 端口的 `!llhd.ref<i1>` 类型，但 `StateType` 的验证器要求类型具有已知的位宽，而 `llhd::RefType` 不在支持的类型列表中。

## Crash Context

- **Tool**: arcilator
- **Dialect**: Arc (使用 LLHD 类型)
- **Failing Pass**: LowerStatePass
- **Crash Type**: Assertion failure

## Error Analysis

### Assertion Message

```
state type must have a known bit width; got '!llhd.ref<i1>'
```

### Key Stack Frames

| Frame | Function | Location |
|-------|----------|----------|
| #12 | `circt::arc::StateType::get(mlir::Type)` | ArcTypes.cpp.inc:108 |
| #13 | `(anonymous namespace)::ModuleLowering::run()` | LowerState.cpp:219 |
| #15 | `(anonymous namespace)::LowerStatePass::runOnOperation()` | LowerState.cpp:1198 |

## Test Case Analysis

### Code Summary

```systemverilog
module MixedPorts(
  input logic clk,
  input logic a,
  output logic b,
  inout wire c    // <-- 问题触发点
);
  logic [7:0] count;
  always_ff @(posedge clk) begin
    if (a) begin
      count <= 8'd0;
    end else begin
      count <= count + 8'd1;
    end
  end
  assign b = count[0];
endmodule
```

### Key Constructs

- `inout wire c` - 双向端口声明
- `always_ff` - 时序逻辑块
- `assign` - 组合逻辑赋值

### Problematic Patterns

`inout wire c` 端口在从 Moore dialect 转换到 Arc dialect 时，被表示为 `!llhd.ref<i1>` 类型。这个引用类型（RefType）不能被包装为 `arc::StateType`，因为 StateType 只支持具有已知位宽的类型。

## CIRCT Source Analysis

### Crash Location

**File**: `lib/Dialect/Arc/Transforms/LowerState.cpp:219`  
**Function**: `ModuleLowering::run()`

### Code Context

```cpp
// Allocate storage for the inputs.
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(arg.getType()), name, storageArg);  // <-- Line 219
  allocatedInputs.push_back(state);
}
```

### StateType Verification Logic

**File**: `lib/Dialect/Arc/ArcTypes.cpp:80-87`

```cpp
LogicalResult
StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Type innerType) {
  if (!computeLLVMBitWidth(innerType))
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

### Supported Types in `computeLLVMBitWidth()`

**File**: `lib/Dialect/Arc/ArcTypes.cpp:29-76`

| Type | Support |
|------|---------|
| `seq::ClockType` | ✅ 1 bit |
| `IntegerType` | ✅ width bits |
| `hw::ArrayType` | ✅ computed |
| `hw::StructType` | ✅ computed |
| `llhd::RefType` | ❌ **NOT SUPPORTED** |

### Processing Path

1. SystemVerilog 模块包含 `inout wire c` 端口
2. `circt-verilog --ir-hw` 输出 HW IR
3. HW IR 中 inout 端口被表示为 `!llhd.ref<i1>` 类型
4. `arcilator` 加载 HW IR 并运行 `LowerStatePass`
5. `LowerStatePass::ModuleLowering::run()` 遍历模块参数
6. 尝试 `StateType::get(arg.getType())` 包装 `!llhd.ref<i1>`
7. `StateType::verify()` 调用 `computeLLVMBitWidth(!llhd.ref<i1>)`
8. **失败** - `llhd::RefType` 不在支持列表中，返回 `std::nullopt`
9. 断言失败

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)

**Cause**: `arcilator` 的 `LowerStatePass` 不支持处理 `inout` 端口类型 (`!llhd.ref<T>`)

**Evidence**:
- 错误消息明确指出 `state type must have a known bit width; got '!llhd.ref<i1>'`
- `computeLLVMBitWidth()` 函数只支持 ClockType、IntegerType、ArrayType、StructType
- `llhd::RefType` 不在支持列表中
- 测例中的 `inout wire c` 端口触发了这个问题

**Mechanism**:
`arcilator` 是 CIRCT 的周期精确模拟器，其 `LowerStatePass` 需要为每个模块输入分配存储空间。`StateType` 要求底层类型具有可计算的位宽以便分配适当的存储。`inout` 端口在 LLHD 中表示为引用类型 (`RefType`)，代表对信号的引用而非值本身，因此没有直接的"位宽"概念。

### Hypothesis 2 (Medium Confidence)

**Cause**: HW IR 到 Arc dialect 的转换管道缺少对 inout 端口的特殊处理

**Evidence**:
- 其他转换（如 HWToSystemC）明确检查并拒绝 inout 端口
- MooreToCore 转换中有注释提到 "inout and ref port is treated"
- `LowerStatePass` 没有在遍历参数前过滤掉 RefType

**Mechanism**:
正确的处理方式应该是：
1. 在 `LowerStatePass` 入口处检测 RefType 参数并发出有意义的错误
2. 或者在更早的 pass 中将 inout 端口转换为不同的表示

## Suggested Fix Directions

### Option 1: 早期错误检测（推荐）

在 `LowerStatePass` 开始处添加检查，如果模块包含不支持的端口类型，发出有意义的诊断信息而不是断言失败：

```cpp
// 在 ModuleLowering::run() 开始处添加
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  if (isa<llhd::RefType>(arg.getType())) {
    return moduleOp.emitError()
        << "arcilator does not support inout/ref ports; port '"
        << moduleOp.getArgName(arg.getArgNumber()) << "' has type "
        << arg.getType();
  }
}
```

### Option 2: 扩展 StateType 支持

在 `computeLLVMBitWidth()` 中添加对 `llhd::RefType` 的支持：

```cpp
if (auto refType = dyn_cast<llhd::RefType>(type)) {
  // RefType 在模拟中可能需要特殊处理
  // 可以考虑使用指针大小或解引用后类型的位宽
  return computeLLVMBitWidth(refType.getNestedType());
}
```

### Option 3: 管道级修复

在 arcilator 管道中添加一个 pass，在 `LowerStatePass` 之前将 inout 端口转换为等效的 input + output 对。

## Keywords for Issue Search

`arcilator` `LowerState` `StateType` `inout` `RefType` `bit width` `LLHD`

## Related Files

| File | Reason |
|------|--------|
| `lib/Dialect/Arc/Transforms/LowerState.cpp` | 崩溃位置，处理模块输入存储分配 |
| `lib/Dialect/Arc/ArcTypes.cpp` | StateType 验证和位宽计算逻辑 |
| `include/circt/Dialect/Arc/ArcTypes.td` | StateType 定义 |
| `include/circt/Dialect/LLHD/IR/LLHDTypes.td` | RefType 定义 |
| `lib/Conversion/MooreToCore/MooreToCore.cpp` | Moore 到 Core 转换中的 inout 处理 |
