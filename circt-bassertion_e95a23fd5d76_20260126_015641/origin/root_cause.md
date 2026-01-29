# Root Cause Analysis Report

## Executive Summary

Arcilator crashes when processing a module with an `inout` port. The `inout wire` port is converted to `!llhd.ref<i1>` type by `circt-verilog --ir-hw`, but arcilator's `LowerState` pass cannot handle LLHD reference types when creating `arc::StateType` for module arguments. The `StateType::verify()` function rejects types without a known bit width, and `!llhd.ref<T>` is not supported by the `computeLLVMBitWidth()` function.

## Crash Context

- **Tool/Command**: `circt-verilog --ir-hw | arcilator`
- **Dialect**: Arc (arcilator), with input from HW/LLHD dialects
- **Failing Pass**: `LowerStatePass` (Arc dialect)
- **Crash Type**: Assertion failure in `StateType::get()`

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
#15 (anonymous namespace)::LowerStatePass::runOnOperation() LowerState.cpp:1198
```

## Test Case Analysis

### Code Summary
The test case defines a simple module `MixedPorts` with:
- An input port `a`
- An output port `b`
- An **inout wire port `c`** (the problematic construct)
- A packed struct type `my_struct_t`
- An `always_comb` block that assigns to struct fields

### Key Constructs
- **`inout wire c`**: This bidirectional port is the root cause of the crash. When lowered through `circt-verilog --ir-hw`, it becomes an `!llhd.ref<i1>` type.
- **Packed struct**: `my_struct_t` with `field1` (4-bit) and `valid` (1-bit)

### Potentially Problematic Patterns
The `inout` port type is not supported by arcilator's state lowering infrastructure. The LLHD reference type (`!llhd.ref<T>`) used to represent bidirectional ports cannot be converted to an `arc::StateType`.

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

The code iterates over all module arguments and creates a `StateType` for each. When `arg.getType()` is `!llhd.ref<i1>`, `StateType::get()` fails.

### StateType Verification Logic
**File**: `lib/Dialect/Arc/ArcTypes.cpp`
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

### computeLLVMBitWidth Function
The function only handles:
- `seq::ClockType` → 1 bit
- `IntegerType` → width from type
- `hw::ArrayType` → recursive computation
- `hw::StructType` → recursive computation

**Missing**: `llhd::RefType` is not handled, causing `computeLLVMBitWidth()` to return `std::nullopt`.

### Processing Path
1. SystemVerilog `inout wire c` is parsed
2. `circt-verilog --ir-hw` converts it to `!llhd.ref<i1>` type
3. Arcilator receives the IR with LLHD reference types
4. `LowerStatePass` tries to allocate storage for all module arguments
5. `StateType::get(!llhd.ref<i1>)` is called
6. `StateType::verify()` calls `computeLLVMBitWidth(!llhd.ref<i1>)`
7. `computeLLVMBitWidth()` returns `std::nullopt` (type not recognized)
8. Assertion fails

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: Arcilator's `LowerState` pass does not support `inout` ports represented as `!llhd.ref<T>` types.

**Evidence**:
- Error message explicitly states: `state type must have a known bit width; got '!llhd.ref<i1>'`
- `computeLLVMBitWidth()` in `ArcTypes.cpp` has no case for `llhd::RefType`
- Stack trace shows crash in `StateType::get()` called from `ModuleLowering::run()`
- Test case contains `inout wire c` which maps to `!llhd.ref<i1>`

**Mechanism**: 
The `LowerState` pass assumes all module arguments can be converted to `arc::StateType`, but LLHD reference types (used for bidirectional ports) are not supported. The pass should either:
1. Handle `llhd::RefType` specially (e.g., extract the nested type)
2. Reject modules with inout ports with a proper error message
3. Add `llhd::RefType` support to `computeLLVMBitWidth()`

### Hypothesis 2 (Medium Confidence)
**Cause**: The `circt-verilog --ir-hw` pipeline should not produce `!llhd.ref<T>` types when the output is intended for arcilator.

**Evidence**:
- Arcilator is designed for simulation, not synthesis
- Inout ports may need different handling in simulation context
- The `--ir-hw` flag suggests HW dialect output, but LLHD types leak through

**Mechanism**:
There may be a missing conversion or lowering step that should transform `!llhd.ref<T>` to a form arcilator can handle before the `LowerState` pass runs.

### Hypothesis 3 (Low Confidence)
**Cause**: The `computeLLVMBitWidth()` function is incomplete and should handle `llhd::RefType`.

**Evidence**:
- The function handles several CIRCT-specific types (`seq::ClockType`, `hw::ArrayType`, `hw::StructType`)
- `llhd::RefType` is a valid CIRCT type that could have a known bit width (the nested type's width)

**Mechanism**:
Adding a case for `llhd::RefType` that extracts and computes the width of the nested type might fix the immediate crash, but may not be semantically correct for simulation purposes.

## Suggested Fix Directions

1. **Add explicit error handling in LowerState pass**: Before calling `StateType::get()`, check if the type is an `llhd::RefType` and emit a proper diagnostic that inout ports are not supported.

2. **Add inout port support to arcilator**: If bidirectional ports should be supported, add proper handling in `LowerState.cpp` to model them appropriately (e.g., as separate read/write states).

3. **Document limitation**: If inout ports are intentionally unsupported, document this limitation and add a check earlier in the arcilator pipeline.

4. **Fix the pipeline**: Ensure `circt-verilog --ir-hw` output intended for arcilator does not contain `!llhd.ref<T>` types, or add a conversion pass.

## Keywords for Issue Search
`arcilator` `inout` `StateType` `llhd.ref` `LowerState` `computeLLVMBitWidth` `bit width`

## Related Files to Investigate
- `lib/Dialect/Arc/Transforms/LowerState.cpp` - Crash location, needs inout handling
- `lib/Dialect/Arc/ArcTypes.cpp` - `computeLLVMBitWidth()` and `StateType::verify()`
- `lib/Dialect/Arc/ArcOps.cpp:339` - Already has check: "inout ports are not supported"
- `tools/arcilator/arcilator.cpp` - Main tool, may need input validation
