# Root Cause Analysis Report

## Executive Summary
Arcilator crashes with an assertion failure when processing SystemVerilog modules that contain `inout` (bidirectional) ports. The `inout` port is lowered to `!llhd.ref<i1>` type, which `arc::StateType` cannot handle because it has no defined bit width computation path.

## Crash Context
- **Tool**: arcilator
- **Dialect**: Arc (during LowerState pass)
- **Failing Pass**: LowerStatePass
- **Crash Type**: Assertion failure
- **CIRCT Version**: 690366b6c (with assertions)

## Error Analysis

### Assertion Message
```
state type must have a known bit width; got '!llhd.ref<i1>'
```

### Key Stack Frames
```
#12 circt::arc::StateType::get(mlir::Type) ArcTypes.cpp.inc:108
#13 (anonymous namespace)::ModuleLowering::run() LowerState.cpp:219
#14 (anonymous namespace)::LowerStatePass::runOnOperation() LowerState.cpp:1198
```

### Crash Location
- **File**: `lib/Dialect/Arc/Transforms/LowerState.cpp`
- **Line**: 219
- **Function**: `ModuleLowering::run()`

## Test Case Analysis

### Code Summary
The test case defines a SystemVerilog module `MixedPorts` with:
- 2 input ports (`a`, `clk`)
- 1 output port (`b`)
- 1 **inout (bidirectional) port** (`c`)
- Internal struct array with `always_ff` logic
- Tri-state assign using `inout` port

### Key Constructs
1. `inout logic c` - bidirectional port declaration
2. `assign c = Qall[0][0].valid ? 1'bz : 1'b0;` - tri-state driver using `inout`

### Problematic Pattern
The `inout` port `c` triggers LLHD reference type generation:
```mlir
hw.module @MixedPorts(in %a : i1, out b : i1, in %c : !llhd.ref<i1>, in %clk : i1)
```

## CIRCT Source Analysis

### Crash Location Code
```cpp
// LowerState.cpp:214-221
// Allocate storage for the inputs.
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(arg.getType()), name, storageArg);  // LINE 219
  allocatedInputs.push_back(state);
}
```

The code iterates over ALL module arguments and attempts to create `StateType` for each, without filtering out types that cannot be represented as states (like `llhd.ref`).

### StateType Verification
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

The `computeLLVMBitWidth` function only handles:
- `seq::ClockType`
- `IntegerType`
- `hw::ArrayType`
- `hw::StructType`

It returns `std::nullopt` for `llhd::RefType`, which causes the assertion to fail.

### Processing Path
1. SystemVerilog `inout logic c` parsed
2. Lowered to HW dialect with `!llhd.ref<i1>` type for bidirectional port
3. Arcilator's LowerState pass tries to allocate state storage for all inputs
4. `StateType::get(!llhd.ref<i1>)` called
5. `StateType::verify()` fails because `computeLLVMBitWidth(!llhd.ref<i1>)` returns `nullopt`
6. Assertion fires in debug build

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence) ⭐
**Cause**: LowerState pass does not filter out `llhd.ref` typed arguments before creating StateType

**Evidence**:
- Code at LowerState.cpp:215-221 processes ALL arguments
- No type check for `llhd::RefType` before `StateType::get()`
- `llhd.ref` represents bidirectional (inout) signals that need special handling

**Mechanism**:
The LowerState pass assumes all module inputs can be represented as simple state storage. Bidirectional (inout) ports are fundamentally different - they represent shared signals that cannot be simply "stored" in arc simulator state.

### Hypothesis 2 (Medium Confidence)
**Cause**: Missing support for `llhd::RefType` in `computeLLVMBitWidth()`

**Evidence**:
- Function only handles Clock, Integer, Array, Struct types
- `llhd::RefType` could potentially be supported if its referenced type is extracted

**Counter-evidence**:
- `llhd::RefType` semantically represents a reference/handle, not a value
- Adding bit width support may be semantically incorrect

### Hypothesis 3 (Low Confidence)
**Cause**: circt-verilog should not emit `llhd.ref` types for use with arcilator

**Evidence**:
- Arcilator is a discrete-event simulator
- LLHD references are part of continuous-time modeling

**Counter-evidence**:
- The HW dialect IR looks reasonable
- This would be a frontend issue, not arcilator issue

## Suggested Fix Directions

### Option 1: Skip or reject llhd.ref inputs in LowerState (Recommended)
Add type checking before creating StateType:
```cpp
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  if (isa<llhd::RefType>(arg.getType())) {
    return moduleOp.emitError("inout ports are not supported by arcilator");
    // Or: continue; // to silently skip
  }
  // ... existing code ...
}
```

### Option 2: Add llhd.ref support to StateType
Add case in `computeLLVMBitWidth`:
```cpp
if (auto refType = dyn_cast<llhd::RefType>(type))
  return computeLLVMBitWidth(refType.getType());
```
Note: This may be semantically incorrect.

### Option 3: Frontend filtering
Ensure `--ir-hw` doesn't emit `llhd.ref` types when targeting arcilator.

## Keywords for Issue Search
`inout` `llhd.ref` `StateType` `LowerState` `arcilator` `bidirectional` `port`

## Related Files
- `lib/Dialect/Arc/Transforms/LowerState.cpp:219`
- `lib/Dialect/Arc/ArcTypes.cpp:78-87`
- `include/circt/Dialect/LLHD/IR/LLHDTypes.td`
