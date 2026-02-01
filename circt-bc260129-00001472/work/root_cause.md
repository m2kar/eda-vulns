# Root Cause Analysis: arcilator StateType assertion failure with `!llhd.ref<i1>` type

## Summary

The arcilator crashes in `LowerState.cpp:219` when processing a hw.module that has an input of type `!llhd.ref<T>` (from SystemVerilog inout ports). The `StateType::get()` function fails the type invariant check because `!llhd.ref<T>` is not a supported type for arc state allocation.

## Error Message

```
error: state type must have a known bit width; got '!llhd.ref<i1>'
```

## Stack Trace Analysis

The crash occurs at:
1. `LowerState.cpp:219` - `ModuleLowering::run()` iterates over module inputs
2. `StateType::get(arg.getType())` is called with `!llhd.ref<i1>` type
3. `ArcTypes.cpp:84` - `StateType::verify()` fails because `computeLLVMBitWidth()` returns nullopt

## Technical Details

### The Input Processing Loop (LowerState.cpp:215-221)

```cpp
// Allocate storage for the inputs.
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(arg.getType()), name, storageArg);  // Line 219
  allocatedInputs.push_back(state);
}
```

This code allocates storage for ALL inputs without checking if their types are supported by the Arc dialect.

### StateType Verification (ArcTypes.cpp:80-87)

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

### computeLLVMBitWidth (ArcTypes.cpp:29-76)

The function only handles:
- `seq::ClockType`
- `IntegerType`
- `hw::ArrayType`
- `hw::StructType`

It does NOT handle `llhd.ref<T>` types, returning `nullopt` for them.

## Trigger Conditions

1. A SystemVerilog module with an `inout` port
2. Using `circt-verilog --ir-hw` to convert to HW dialect
3. The inout port becomes `!llhd.ref<T>` type in HW IR
4. Running through arcilator, which cannot handle ref types

## IR Analysis

The input SystemVerilog:
```systemverilog
module MixedPorts(
  input  logic a,
  input  logic clk,
  output logic b,
  inout  logic c  // <-- This causes the issue
);
```

Generates HW IR with:
```mlir
hw.module @MixedPorts(in %a : i1, in %clk : i1, out b : i1, in %c : !llhd.ref<i1>) {
  ...
}
```

The `in %c : !llhd.ref<i1>` is the problematic port.

## Root Cause Classification

**Type**: Missing input validation / Type handling gap

The arcilator's `LowerState` pass doesn't validate that all module input types are supported before attempting to allocate storage. The `!llhd.ref<T>` type represents a reference/inout port in the LLHD dialect, but the Arc dialect's state allocation mechanism doesn't support reference types.

## Possible Fixes

1. **Early validation**: Add a check in `LowerState` pass to emit a proper diagnostic when encountering unsupported types like `!llhd.ref<T>`

2. **Extend computeLLVMBitWidth**: Handle `llhd.ref<T>` types by computing the bit width of the underlying type T

3. **Graceful error**: Convert the assertion failure to a user-friendly error message before reaching the assert

## Analysis Data

```json
{
  "dialect": "arc",
  "crash_type": "assertion_failure",
  "crash_location": "LowerState.cpp:219",
  "root_cause": "StateType::get() called with unsupported !llhd.ref<T> type",
  "trigger_construct": "SystemVerilog inout port",
  "missing_handler": "llhd.ref type in computeLLVMBitWidth()",
  "severity": "high",
  "classification": "report"
}
```
