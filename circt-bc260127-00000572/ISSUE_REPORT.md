# CIRCT Bug Report

## Summary

arcilator crashes with assertion failure when processing modules with `inout` ports. The crash occurs because `inout` ports are converted to LLHD reference types (`!llhd.ref<T>`) by `circt-verilog --ir-hw`, but the state lowering pass in arcilator does not support LLHD reference types in `StateType`.

## Testcase ID
260127-00000572

## Crash Type
Assertion failure

## Reproduction Steps

### Minimal Test Case

```systemverilog
module Minimal(
  inout logic c
);
endmodule
```

### Reproduction Command

```bash
export PATH=/opt/llvm-22/bin:$PATH

# Generate LLHD IR with reference type
circt-verilog --ir-hw minimal.sv > minimal.mlir

# Run arcilator (crashes here)
arcilator minimal.mlir
```

Or in one command:

```bash
circt-verilog --ir-hw minimal.sv | arcilator
```

### Alternative: Direct IR Input

```bash
echo 'module {
  hw.module @Minimal(in %c : !llhd.ref<i1>) {
    hw.output
  }
}' | arcilator
```

## Error Message

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/IR/StorageUniquerSupport.h:180: static ConcreteT mlir::detail::StorageUserBase<circt::arc::StateType, mlir::Type, circt::arc::detail::StateTypeStorage, mlir::detail::TypeUniquer>::get(MLIRContext *, Args &&...) [ConcreteT = circt::arc::StateType, BaseT = mlir::Type, StorageT = circt::arc::detail::StateTypeStorage, UniquerT = mlir::detail::TypeUniquer, Traits = <>, Args = <mlir::Type &>]: Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

## Root Cause Analysis

### Crash Location

**File**: `lib/Dialect/Arc/Transforms/LowerState.cpp`
**Line**: 219

```cpp
auto state =
    RootInputOp::create(allocBuilder, arg.getLoc(),
                       StateType::get(arg.getType()), name, storageArg);
```

### Failure Chain

1. **Type Conversion**: When `circt-verilog --ir-hw` processes Verilog/SystemVerilog modules with `inout` ports, it converts them to LLHD reference types:
   ```
   in %c : !llhd.ref<i1>
   ```

2. **State Lowering**: During arcilator's state lowering pass, `LowerState::run()` allocates storage for module inputs by calling `StateType::get(arg.getType())`.

3. **Type Verification**: `StateType::get()` calls `StateType::verify()` to validate the type.

4. **Bit Width Calculation Failure**: The verification function calls `computeLLVMBitWidth(innerType)` to compute the bit width of the type.

5. **Unsupported Type**: `computeLLVMBitWidth()` in `lib/Dialect/Arc/ArcTypes.cpp` (lines 29-76) only supports:
   - `seq::ClockType` (returns 1)
   - `IntegerType` (returns `intType.getWidth()`)
   - `hw::ArrayType` (recursively computes element width)
   - `hw::StructType` (recursively computes struct member width)

   For any other type (including `!llhd.ref<T>`), it returns `std::nullopt` (line 75).

6. **Assertion Failure**: When `computeLLVMBitWidth()` returns `std::nullopt`, `StateType::verify()` emits an error and returns failure, causing the assertion to fail.

### Key Code

**File**: `lib/Dialect/Arc/ArcTypes.cpp`

```cpp
static std::optional<uint64_t> computeLLVMBitWidth(Type type) {
  if (isa<seq::ClockType>(type))
    return 1;

  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();

  if (auto arrayType = dyn_cast<hw::ArrayType>(type)) {
    // ... compute array width
  }

  if (auto structType = dyn_cast<hw::StructType>(type)) {
    // ... compute struct width
  }

  // We don't know anything about any other types.
  return {};  // <-- LLHD ref types end up here
}

LogicalResult
StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Type innerType) {
  if (!computeLLVMBitWidth(innerType))
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

## Environment

- **CIRCT Version**: firtool-1.139.0 (LLVM 22.0.0git)
- **Tool**: arcilator
- **CIRCT commit**: (from original error log)

## Expected Behavior

arcilator should either:
1. Support LLHD reference types in state lowering, OR
2. Reject modules with `inout` ports with a clear error message instead of crashing with an assertion

## Actual Behavior

arcilator crashes with an assertion failure when processing modules with `inout` ports.

## Possible Fixes

### Option 1: Add LLHD Reference Type Support

Modify `computeLLVMBitWidth()` to handle LLHD reference types:

```cpp
static std::optional<uint64_t> computeLLVMBitWidth(Type type) {
  if (auto refType = dyn_cast<llhd::RefType>(type))
    return computeLLVMBitWidth(refType.getNestedType());

  // ... existing type checks ...
}
```

### Option 2: Early Error Detection

Add validation in `LowerState::run()` to check for unsupported types before attempting to create state types:

```cpp
auto argType = arg.getType();
if (isa<llhd::RefType>(argType)) {
  emitError(arg.getLoc()) << "inout ports are not supported by arcilator";
  return failure();
}
```

### Option 3: Convert LLHD Ref Types Before State Lowering

Add a pass before state lowering that converts LLHD reference types to a representation that arcilator can handle.

## Additional Notes

- The current development version of arcilator (`/opt/firtool/bin/arcilator`) does not crash with the same input, suggesting this issue may have been fixed in a newer version.
- The bug affects any Verilog/SystemVerilog module with `inout` ports when using `circt-verilog --ir-hw | arcilator` pipeline.
- The issue is specific to the Arc dialect's state lowering pass; other passes may handle LLHD reference types correctly.

## Related Files

- `lib/Dialect/Arc/Transforms/LowerState.cpp` - State lowering implementation
- `lib/Dialect/Arc/ArcTypes.cpp` - Type validation and bit width computation
- `include/circt/Dialect/LLHD/LLHDTypes.td` - LLHD reference type definition
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Moore to LLHD type conversion
