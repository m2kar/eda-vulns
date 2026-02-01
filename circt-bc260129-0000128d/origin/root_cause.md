# Root Cause Analysis Report

**Testcase ID**: 260129-0000128d  
**Crash Type**: Assertion Failure  
**Affected Tool**: arcilator  
**Dialect**: Arc (with LLHD type)

## Summary

The arcilator crashes with an assertion failure when processing a SystemVerilog module containing an `inout` (bidirectional) port. The crash occurs because the LLHD reference type (`!llhd.ref<i1>`) used to represent bidirectional ports is not supported by the Arc dialect's `StateType`, which requires types with known bit widths.

## Error Analysis

### Error Message
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
```

### Assertion Failure
```
Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Stack Trace (Key Frames)
1. `StateType::get(mlir::Type)` - ArcTypes.cpp.inc:108
2. `ModuleLowering::run()` - LowerState.cpp:219
3. `LowerStatePass::runOnOperation()` - LowerState.cpp:1198

## Code Analysis

### Problematic SystemVerilog Construct
```systemverilog
module MixPorts(
  input logic clk,
  input logic rst,
  input logic a,
  output logic b,
  inout wire c    // <-- This bidirectional port causes the crash
);
  // ...
  assign c = a ? 1'bz : 1'b0;  // Tristate assignment
endmodule
```

The module uses an `inout wire` port with a tristate assignment (`1'bz`), which is valid SystemVerilog but not supported by the arcilator simulation flow.

## Root Cause Hypothesis

### Primary Cause
The `LowerStatePass` in arcilator attempts to allocate storage for all module input arguments, including bidirectional ports. The relevant code in `LowerState.cpp` (line ~219) iterates over all block arguments:

```cpp
// Allocate storage for the inputs.
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state = RootInputOp::create(allocBuilder, arg.getLoc(),
                                   StateType::get(arg.getType()), // <-- Crash here
                                   name, storageArg);
  allocatedInputs.push_back(state);
}
```

When the `inout` port is encountered, its type is `!llhd.ref<i1>` (LLHD reference type used for bidirectional signals). The `StateType::get()` call triggers verification which checks if the inner type has a known bit width via `computeLLVMBitWidth()`:

```cpp
// In ArcTypes.cpp
LogicalResult StateType::verify(..., Type innerType) {
  if (!computeLLVMBitWidth(innerType))
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

The `computeLLVMBitWidth()` function only handles:
- `seq::ClockType`
- `IntegerType`
- `hw::ArrayType`
- `hw::StructType`

It does **not** handle `llhd::RefType`, causing the verification to fail.

### Contributing Factors
1. The CIRCT frontend (`circt-verilog`) correctly parses the `inout` port and represents it using LLHD reference semantics
2. However, the arcilator's state lowering pass assumes all input types can be represented as Arc `StateType`
3. There's no early validation or graceful error handling for unsupported port types before the lowering pass

## Suggested Fix

### Option 1: Early Validation (Recommended)
Add validation in the LowerStatePass to detect and reject unsupported port types with a meaningful error message:

```cpp
// Before allocating inputs
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  if (isa<llhd::RefType>(arg.getType())) {
    return moduleOp.emitOpError()
        << "bidirectional (inout) ports are not supported by arcilator; "
        << "port '" << moduleOp.getArgName(arg.getArgNumber()) << "' has type "
        << arg.getType();
  }
}
```

### Option 2: Extend computeLLVMBitWidth
If arcilator should support reference types in some capacity, extend `computeLLVMBitWidth()` to handle `llhd::RefType` by extracting the underlying type:

```cpp
if (auto refType = dyn_cast<llhd::RefType>(type))
  return computeLLVMBitWidth(refType.getNestedType());
```

However, this would require additional changes to properly handle the semantics of bidirectional signals in simulation.

## Reproduction

### Minimal Test Case
```systemverilog
module Test(inout wire c);
  assign c = 1'bz;
endmodule
```

### Command
```bash
circt-verilog --ir-hw test.sv | arcilator
```

## Impact Assessment

- **Severity**: Medium - Causes crash instead of graceful error
- **Scope**: Any SystemVerilog design using `inout` ports processed through arcilator
- **Workaround**: Avoid using `inout` ports in designs intended for arcilator simulation

## References

- `lib/Dialect/Arc/Transforms/LowerState.cpp` - State lowering pass
- `lib/Dialect/Arc/ArcTypes.cpp` - StateType verification
- `include/circt/Dialect/LLHD/IR/LLHDTypes.td` - LLHD RefType definition
