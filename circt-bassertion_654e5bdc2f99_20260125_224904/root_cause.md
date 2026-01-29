# Root Cause Analysis Report

## Executive Summary

arcilator crashes when processing SystemVerilog modules with `inout` ports. The crash occurs in `LowerStatePass` when it attempts to create `arc::StateType` for an `llhd.ref<i6>` type argument (representing the inout port). The `StateType::verify()` function fails because `computeLLVMBitWidth()` does not support `llhd::RefType`.

## Crash Context

- **Tool/Command**: `arcilator` (via pipeline: `circt-verilog --ir-hw | arcilator`)
- **Dialect**: Arc
- **Failing Pass**: `LowerStatePass`
- **Crash Type**: Assertion failure
- **CIRCT Version**: 1.139.0

## Error Analysis

### Assertion/Error Message
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i6>'
arcilator: ...mlir/include/mlir/IR/StorageUniquerSupport.h:180: Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Key Stack Frames
```
#11 0x000055711bab449c (arcilator)
#12 circt::arc::StateType::get(mlir::Type) ArcTypes.cpp.inc:108
#13 (anonymous namespace)::ModuleLowering::run() LowerState.cpp:219
#14 (anonymous namespace)::LowerStatePass::runOnOperation() LowerState.cpp:1198
```

## Test Case Analysis

### Code Summary
The test case `source.sv` defines a module `MixedPorts` with:
- Two signed 6-bit inputs (`a`, `b`)
- One signed 6-bit output (`c`)
- One signed 6-bit **inout** port (`d`)

The module performs an arithmetic signed right shift and conditionally drives the inout port.

### Key Constructs
- `inout` port declaration (`inout signed [5:0] d`)
- Conditional assignment to inout port (`assign d = (b[0]) ? c : a`)
- Signed arithmetic right shift (`a >>> b`)

### Potentially Problematic Patterns
The `inout` port is the trigger for this crash. When `circt-verilog` lowers SystemVerilog to HW IR, the inout port becomes an argument with `llhd.ref<i6>` type.

## CIRCT Source Analysis

### Crash Location
**File**: `lib/Dialect/Arc/Transforms/LowerState.cpp`
**Function**: `ModuleLowering::run()`
**Line**: 219

### Code Context
```cpp
// Allocate storage for the inputs.
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(arg.getType()), name, storageArg);
  allocatedInputs.push_back(state);
}
```

The loop iterates over **all** module arguments without filtering by port direction. For inout ports, `arg.getType()` returns `llhd.ref<i6>`, which is passed to `StateType::get()`.

### Validation in StateType
```cpp
// lib/Dialect/Arc/ArcTypes.cpp:80-87
LogicalResult
StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Type innerType) {
  if (!computeLLVMBitWidth(innerType))
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

### computeLLVMBitWidth Support
The function `computeLLVMBitWidth()` only supports:
- `seq::ClockType`
- `IntegerType`
- `hw::ArrayType`
- `hw::StructType`

It does **not** support `llhd::RefType`, causing the validation failure.

### Existing Inout Port Check
There is an existing check in `ModelOp::verify()`:
```cpp
// lib/Dialect/Arc/ArcOps.cpp:337-339
for (const hw::ModulePort &port : getIo().getPorts())
  if (port.dir == hw::ModulePort::Direction::InOut)
    return emitOpError("inout ports are not supported");
```

However, this check only runs after `ModelOp` is created, and the crash occurs earlier during `LowerStatePass`.

### Processing Path
1. `circt-verilog --ir-hw` converts SystemVerilog to HW IR
2. Inout port becomes an argument with `llhd.ref<T>` type
3. Output is piped to `arcilator`
4. `arcilator` runs `LowerStatePass`
5. `ModuleLowering::run()` iterates all module arguments
6. For the inout port argument: calls `StateType::get(llhd.ref<i6>)`
7. `StateType::verify()` calls `computeLLVMBitWidth(llhd.ref<i6>)`
8. `computeLLVMBitWidth()` returns `std::nullopt` (type not supported)
9. Assertion fails

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: `LowerState.cpp` lacks filtering or early rejection of `llhd.ref` types when iterating module arguments.

**Evidence**:
- The crash occurs at line 219 when iterating all arguments
- No port direction check exists in the loop
- `computeLLVMBitWidth()` explicitly does not support `llhd::RefType`
- Error message shows `llhd.ref<i6>` being passed to `StateType::get()`

**Mechanism**: 
The `ModuleLowering::run()` function iterates over all module body arguments and creates `RootInputOp` for each. It assumes all arguments have types that can be converted to `StateType`. When an inout port is present, its `llhd.ref` type fails the `StateType` validation.

### Hypothesis 2 (Medium Confidence)
**Cause**: The pipeline lacks a preprocessing step to reject or transform modules with inout ports before they reach `LowerStatePass`.

**Evidence**:
- `ModelOp::verify()` checks for inout ports but runs too late
- There's no earlier pipeline stage that validates port compatibility
- The preprocessing pipeline (`ArcPreprocessingPipeline`) doesn't filter inout ports

**Mechanism**:
While `ModelOp::verify()` would reject inout ports, the crash occurs during the lowering phase before `ModelOp` is fully constructed and verified.

### Hypothesis 3 (Low Confidence)
**Cause**: Missing support for `llhd::RefType` in `computeLLVMBitWidth()`.

**Evidence**:
- The function supports limited types
- `llhd::RefType` is not in the supported list

**Counter-evidence**:
- `llhd::RefType` represents a reference/pointer, not a value type
- Adding bit width support for references doesn't make semantic sense
- This is more of a "how arcilator could handle it" rather than the root cause

## Suggested Fix Directions

1. **Early Rejection in LowerStatePass** (Recommended):
   Add a check in `ModuleLowering::run()` to skip or error on arguments with `llhd.ref` types:
   ```cpp
   for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
     if (isa<llhd::RefType>(arg.getType())) {
       return moduleOp.emitError("inout ports are not supported in arcilator");
     }
     // ... existing code
   }
   ```

2. **Pipeline Validation Pass**:
   Add a preprocessing pass that validates all modules are arcilator-compatible before lowering begins.

3. **Better Error Message**:
   At minimum, emit a diagnostic before the assertion fires, explaining that inout ports are not supported.

## Keywords for Issue Search
`arcilator` `inout` `LowerState` `StateType` `llhd.ref` `bit width` `assertion`

## Related Files to Investigate
- `lib/Dialect/Arc/Transforms/LowerState.cpp` - Crash location, needs port direction check
- `lib/Dialect/Arc/ArcTypes.cpp` - `StateType::verify()` and `computeLLVMBitWidth()`
- `lib/Dialect/Arc/ArcOps.cpp` - Existing `ModelOp::verify()` inout check
- `tools/arcilator/arcilator.cpp` - Pipeline configuration
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Inout port type conversion to `llhd.ref`
