# Root Cause Analysis Report

## Executive Summary

Arcilator crashes when processing a SystemVerilog module with an `inout` port. The `inout` port is converted to `llhd.ref<i1>` type by `circt-verilog --ir-hw`, but `arcilator`'s `LowerStatePass` cannot handle LLHD RefType when creating state storage. The `computeLLVMBitWidth()` function in Arc dialect doesn't support RefType, causing `StateType::verify()` to fail with an assertion.

## Crash Context

- **Tool/Command**: `circt-verilog --ir-hw | arcilator`
- **Dialect**: Arc (with LLHD types in input)
- **Failing Pass**: `LowerStatePass`
- **Crash Type**: Assertion failure

## Error Analysis

### Assertion/Error Message
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Key Stack Frames
```
#12 circt::arc::StateType::get(mlir::Type) ArcTypes.cpp.inc:108
#13 (anonymous namespace)::ModuleLowering::run() LowerState.cpp:219
#15 (anonymous namespace)::LowerStatePass::runOnOperation() LowerState.cpp:1198
```

## Test Case Analysis

### Code Summary
A simple SystemVerilog module `MixedPorts` with mixed port types:
- Input ports: `clk`, `a`
- Output port: `b`
- **Inout port**: `c` (the problematic construct)

The module assigns `c = a` (continuous assignment to inout) and has a clocked register `b <= a`.

### Key Constructs
- **`inout logic c`**: Bidirectional port converted to `llhd.ref<i1>` in HW IR
- `always_ff @(posedge clk)`: Standard clocked process
- Continuous assignment to inout port

### Potentially Problematic Patterns
The **inout port** is the root cause. SystemVerilog `inout` ports represent bidirectional signals which are converted to LLHD's reference types (`llhd.ref<T>`) in the HW IR output.

## CIRCT Source Analysis

### Crash Location
**File**: `lib/Dialect/Arc/Transforms/LowerState.cpp`
**Function**: `ModuleLowering::run()`
**Line**: 219

### Code Context
```cpp
// LowerState.cpp:214-221
// Allocate storage for the inputs.
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(arg.getType()), name, storageArg);
  allocatedInputs.push_back(state);
}
```

The pass iterates over all module arguments (ports) and creates `RootInputOp` for each. It calls `StateType::get(arg.getType())` which internally calls `StateType::verify()`.

### Type Verification Logic
```cpp
// ArcTypes.cpp:80-87
LogicalResult
StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Type innerType) {
  if (!computeLLVMBitWidth(innerType))
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}
```

### Bit Width Computation
```cpp
// ArcTypes.cpp:29-76
static std::optional<uint64_t> computeLLVMBitWidth(Type type) {
  if (isa<seq::ClockType>(type))
    return 1;
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (auto arrayType = dyn_cast<hw::ArrayType>(type)) {
    // ... handles hw::ArrayType
  }
  if (auto structType = dyn_cast<hw::StructType>(type)) {
    // ... handles hw::StructType
  }
  // We don't know anything about any other types.
  return {};
}
```

**Note**: `llhd::RefType` is NOT handled in `computeLLVMBitWidth()`.

### Processing Path
1. `circt-verilog --ir-hw` parses SystemVerilog and produces HW IR
2. The `inout` port becomes an argument with type `!llhd.ref<i1>`
3. `arcilator` runs `LowerStatePass` on the HW module
4. `ModuleLowering::run()` tries to allocate storage for each module argument
5. For the inout port, it calls `StateType::get(!llhd.ref<i1>)`
6. `StateType::verify()` calls `computeLLVMBitWidth(!llhd.ref<i1>)`
7. `computeLLVMBitWidth()` returns `{}` (no handler for RefType)
8. Verification fails, assertion triggers

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: Arc dialect's `StateType` doesn't support LLHD `RefType` because `computeLLVMBitWidth()` only handles basic types (`IntegerType`, `seq::ClockType`, `hw::ArrayType`, `hw::StructType`).

**Evidence**:
- Error message explicitly says: `state type must have a known bit width; got '!llhd.ref<i1>'`
- `computeLLVMBitWidth()` source code has no case for `llhd::RefType`
- Test case uses `inout` port which maps to `llhd.ref<i1>`
- Stack trace shows crash in `StateType::get()` → `StateType::verify()`

**Mechanism**: 
When `arcilator` processes a module with an `inout` port (represented as `llhd.ref<T>`), the `LowerStatePass` tries to create state storage for all inputs. The `StateType::get()` verifier checks if the type has a known bit width via `computeLLVMBitWidth()`, but this function doesn't handle `llhd::RefType`, so it returns `{}`, causing the assertion to fail.

### Hypothesis 2 (Medium Confidence)
**Cause**: `inout` ports are fundamentally unsupported by arcilator and should be rejected earlier in the pipeline with a proper error message.

**Evidence**:
- Arcilator is a simulator focused on cycle-based simulation
- `inout` ports represent bidirectional wires which are difficult to simulate in this model
- No graceful error handling exists for this case

**Mechanism**:
The frontend (`circt-verilog --ir-hw`) accepts `inout` ports and produces LLHD ref types, but arcilator doesn't have proper support or error handling for them.

## Suggested Fix Directions

1. **Add RefType support to `computeLLVMBitWidth()`**: Extract the nested type from `llhd::RefType` and compute its bit width. This would allow arcilator to handle inout ports.

2. **Add early validation**: Add a pre-pass check in arcilator to reject modules with `llhd::RefType` ports with a proper error message instead of an assertion failure.

3. **Document limitation**: If inout ports are intentionally unsupported, document this limitation clearly.

## Keywords for Issue Search
`arcilator` `inout` `llhd.ref` `StateType` `LowerState` `computeLLVMBitWidth` `RefType`

## Related Files to Investigate
- `lib/Dialect/Arc/ArcTypes.cpp` - Add RefType handling to `computeLLVMBitWidth()`
- `lib/Dialect/Arc/Transforms/LowerState.cpp` - Crash site, may need validation
- `include/circt/Dialect/LLHD/IR/LLHDTypes.td` - RefType definition
- `tools/arcilator/arcilator.cpp` - Entry point, could add validation
