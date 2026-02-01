# Root Cause Analysis Report

## Executive Summary

Arcilator crashes when attempting to simulate a SystemVerilog module containing `inout` (bidirectional) ports. The `LowerState` pass fails to handle `!llhd.ref<T>` types generated from inout ports, as `StateType::verify()` only supports types with a known bit width computable via `computeLLVMBitWidth()`.

## Crash Context

- **Tool/Command**: `arcilator` (via pipeline: `circt-verilog --ir-hw | arcilator`)
- **Dialect**: Arc (with LLHD reference types from inout ports)
- **Failing Pass**: `LowerStatePass` (`arc-lower-state`)
- **Crash Type**: Assertion failure

## Error Analysis

### Assertion/Error Message

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
Assertion `succeeded(ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Key Stack Frames

```
#11 0x000055f984b07bbc StateType::verify() - ArcTypes.cpp.inc:108
#12 0x000055f984b07ae9 circt::arc::StateType::get(mlir::Type) - ArcTypes.cpp.inc:108
#13 0x000055f984b72f5c ModuleLowering::run() - LowerState.cpp:219
#14 0x000055f984b72f5c LowerStatePass::runOnOperation() - LowerState.cpp:1198
```

## Test Case Analysis

### Code Summary

The test case is a simple SystemVerilog module with mixed port types including an `inout` (bidirectional) port used for tri-state logic:

```systemverilog
module MixedPorts(
  input logic clk,
  input logic a,
  output logic b,
  inout logic c        // <-- Problematic bidirectional port
);
  logic [3:0] counter = 0;
  
  always_ff @(posedge clk) begin
    if (a) begin
      counter <= counter + 1;
    end
  end
  
  assign b = (counter == 4'b1111);
  assign c = (counter[0]) ? 1'b1 : 1'bz;  // Tri-state assignment
endmodule
```

### Key Constructs

| Construct | Relevance to Crash |
|-----------|-------------------|
| `inout logic c` | **Direct cause** - bidirectional port converted to `!llhd.ref<i1>` |
| `1'bz` tri-state | Requires bidirectional signal handling |
| `always_ff` | Not directly related but part of state lowering |

### Potentially Problematic Patterns

The `inout` port declaration creates a bidirectional signal that cannot be represented as a simple storage state in arcilator. When `circt-verilog --ir-hw` compiles this, the inout port becomes an `!llhd.ref<i1>` type (LLHD signal reference), which the Arc dialect's state allocation cannot handle.

## CIRCT Source Analysis

### Crash Location

**File**: `lib/Dialect/Arc/ArcTypes.cpp`  
**Function**: `StateType::verify()`  
**Line**: ~79-83

### Code Context

```cpp
// From lib/Dialect/Arc/ArcTypes.cpp

static std::optional<uint64_t> computeLLVMBitWidth(Type type) {
  if (isa<seq::ClockType>(type))
    return 1;

  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();

  if (auto arrayType = dyn_cast<hw::ArrayType>(type)) {
    // ... handles arrays
  }

  if (auto structType = dyn_cast<hw::StructType>(type)) {
    // ... handles structs
  }

  // We don't know anything about any other types.
  return {};  // <-- !llhd.ref<i1> falls through here
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

The crash path in `LowerState.cpp`:

```cpp
// From lib/Dialect/Arc/Transforms/LowerState.cpp:219 (ModuleLowering::run)

// Allocate storage for the inputs.
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(arg.getType()), // <-- Crash here for inout
                          name, storageArg);
  allocatedInputs.push_back(state);
}
```

### Processing Path

1. `circt-verilog --ir-hw` parses the SystemVerilog and produces HW+LLHD IR
2. The `inout logic c` port is converted to `!llhd.ref<i1>` type
3. `arcilator` runs its pass pipeline including `LowerStatePass`
4. `ModuleLowering::run()` iterates over module arguments to allocate input storage
5. For the inout port argument with type `!llhd.ref<i1>`, it calls `StateType::get(arg.getType())`
6. `StateType::get()` calls `verifyInvariants()` which calls `StateType::verify()`
7. `StateType::verify()` calls `computeLLVMBitWidth(!llhd.ref<i1>)`
8. `computeLLVMBitWidth()` doesn't handle `llhd::RefType`, returns `std::nullopt`
9. Verification fails, assertion triggers, crash

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)

**Cause**: `inout` ports are not supported by arcilator's `LowerState` pass. The pass assumes all module port types can be wrapped in `StateType`, but `!llhd.ref<T>` types from bidirectional ports cannot have their bit width computed.

**Evidence**:
- Error message explicitly states `!llhd.ref<i1>` has no known bit width
- `computeLLVMBitWidth()` in `ArcTypes.cpp` has no case for `llhd::RefType`
- The test case contains an `inout` port which is the only non-standard port type
- Arcilator is designed for simulation where bidirectional signals require special handling

**Mechanism**: The `LowerState` pass blindly iterates over all module arguments and tries to create `StateType` storage for each. For `inout` ports represented as `!llhd.ref<T>`, this fails because `StateType` requires a type with computable bit width.

### Hypothesis 2 (Medium Confidence)

**Cause**: Missing frontend validation for unsupported features. Arcilator should reject modules with `inout` ports earlier in the pipeline with a clear error message instead of crashing.

**Evidence**:
- No error message before the assertion indicates this is an unexpected code path
- The crash happens deep in type verification rather than during input validation
- Similar tools typically emit "unsupported feature" errors for bidirectional ports

## Suggested Fix Directions

1. **Add `inout` port detection and rejection**: In `LowerStatePass::runOnOperation()` or earlier, check for `inout` ports and emit a proper error message like "arcilator does not support bidirectional (inout) ports".

2. **Extend `computeLLVMBitWidth()` for reference types**: If arcilator should support bidirectional ports in the future, add handling for `llhd::RefType` by computing the bit width of the inner type.

3. **Add module-level validation pass**: Create or extend a validation pass that runs before `LowerState` to check for unsupported constructs.

## Keywords for Issue Search

`arcilator` `inout` `bidirectional` `llhd.ref` `StateType` `LowerState` `computeLLVMBitWidth` `state type must have a known bit width`

## Related Files to Investigate

| File | Reason |
|------|--------|
| `lib/Dialect/Arc/ArcTypes.cpp` | `computeLLVMBitWidth()` needs extension for ref types |
| `lib/Dialect/Arc/Transforms/LowerState.cpp` | Main crash site, needs inout port handling |
| `include/circt/Dialect/Arc/ArcTypes.td` | `StateType` definition with verification |
| `tools/arcilator/arcilator.cpp` | Tool entry point, could add early validation |
