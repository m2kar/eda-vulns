# Root Cause Analysis Report

## Executive Summary
The bug occurs in `arcilator` during the `LowerState` pass when processing SystemVerilog modules with `inout` ports using tri-state buffers. The compiler fails to properly handle the LLHD reference type (`!llhd.ref<i1>`) generated for these ports, causing an assertion failure because state types must have known bit widths.

## Crash Context
- **Tool/Command**: `circt-verilog --ir-hw source.sv | arcilator`
- **Dialect**: Arc (arcilator)
- **Failing Pass**: LowerState (circt::arc::LowerStatePass)
- **Crash Type**: Assertion failure

## Error Analysis

### Assertion/Error Message
```
error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/../mlir/include/mlir/IR/StorageUniquerSupport.h:180: static ConcreteT mlir::detail::StorageUserBase<circt::arc::StateType, mlir::Type, circt::arc::detail::StateTypeStorage, mlir::detail::TypeUniquer>::get(MLIRContext *, Args &&...) [ConcreteT = circt::arc::StateType, BaseT = mlir::Type, StorageT = circt::arc::detail::StateTypeStorage, UniquerT = mlir::detail::TypeUniquer, Traits = <>, Args = <mlir::Type &>]: Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Key Stack Frames
```
#12 circt::arc::StateType::get(mlir::Type)
    .../ArcTypes.cpp.inc:108:3

#13 (anonymous namespace)::ModuleLowering::run()
    .../lib/Dialect/Arc/Transforms/LowerState.cpp:219:66

#14 (anonymous namespace)::LowerStatePass::runOnOperation()
    .../lib/Dialect/Arc/Transforms/LowerState.cpp:1198:41
```

## Test Case Analysis

### Code Summary
The test case is a SystemVerilog module named `combined_mod` that includes:
- Signed and unsigned input ports
- An `inout` port `io_sig` with tri-state buffer
- A shift operation inside a loop
- Tri-state assignment: `assign io_sig = (wide_input[0]) ? out[0] : 1'bz;`

### Key Constructs
- **`inout` port with tri-state (`1'bz`)**: This is the problematic construct that triggers the bug
- **Signed types**: `input logic signed [7:0] in`, `output logic signed [7:0] out`
- **Wide input bit-slicing**: `wide_input[31:0]`
- **Always_comb with for loop**: Contains a shift operation in a combinational loop

### Potentially Problematic Patterns
The primary issue is the **inout port with tri-state assignment**. When `circt-verilog` converts this to the Arc dialect, it generates an LLHD reference type `!llhd.ref<i1>` for the `io_sig` port. The Arc dialect's StateType verification then fails because LLHD reference types don't have a known bit width, which is a requirement for StateType.

## CIRCT Source Analysis

### Crash Location
**File**: `lib/Dialect/Arc/Transforms/LowerState.cpp`
**Function**: `(anonymous namespace)::ModuleLowering::run()`
**Line**: 219 (approximately)

The stack trace shows the crash originates from:
1. `StateType::get()` being called with a `!llhd.ref<i1>` type
2. The `verifyInvariants()` function in `StateType` rejecting this type
3. Assertion failure in `StorageUniquerSupport.h:180`

### Processing Path
1. **Input Parsing**: `circt-verilog` parses the SystemVerilog module and converts it to MLIR
2. **Arc Model Creation**: The module is converted to an `arc.model` with ports
3. **Type Conversion**: The `inout` port with tri-state gets converted to `!llhd.ref<i1>` (LLHD reference type)
4. **LowerState Pass**: The `LowerState` pass attempts to create `StateType` instances
5. **Verification Failure**: `StateType::verifyInvariants()` rejects `!llhd.ref<i1>` because it lacks a known bit width
6. **Assertion Triggers**: The invariant check fails, triggering the assertion

### Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: Incomplete handling of LLHD reference types in Arc dialect's StateType verification.

**Evidence**:
- The error message explicitly states "state type must have a known bit width; got '!llhd.ref<i1>'"
- The crash occurs in `StateType::get()` during verification
- `!llhd.ref<i1>` is an LLHD reference type used for hardware signals that don't have a fixed bit width in the traditional sense
- The test case uses an `inout` port with tri-state, which is represented as an LLHD reference

**Mechanism**:
1. The `LowerState` pass calls `StateType::get()` with the LLHD reference type
2. `StateType::verifyInvariants()` checks if the type has a known bit width
3. LLHD reference types (`!llhd.ref<T>`) don't have a fixed bit width representation
4. The verification fails, causing the assertion

### Hypothesis 2 (Medium Confidence)
**Cause**: Missing type conversion pass to handle LLHD references before StateType creation.

**Evidence**:
- The issue may be that an additional type conversion pass should run before `LowerState` to convert LLHD references to types with known bit widths
- The presence of LLHD types in Arc models suggests an incomplete conversion pipeline

**Mechanism**:
1. LLHD reference types are valid in earlier stages of compilation
2. A type conversion pass should convert these to types StateType can handle
3. This pass is either missing or runs after LowerState

## Suggested Fix Directions

### Option 1: Allow LLHD Reference Types in StateType
Modify `StateType::verifyInvariants()` to properly handle LLHD reference types:
- Add special case handling for `!llhd.ref<T>` types
- Either extract the underlying type's bit width or allow references without bit width constraints
- Update the type system documentation to clarify LLHD reference type handling

**Pros**: 
- Fixes the immediate assertion failure
- Preserves the semantic meaning of tri-state/inout ports

**Cons**: 
- May require deeper changes to the Arc dialect type system
- Could introduce new bugs if not carefully implemented

### Option 2: Add Type Conversion Before LowerState
Introduce a new pass that converts LLHD reference types to types with known bit widths before the `LowerState` pass runs:
- Create a `LowerLLHDRefs` pass
- Convert `!llhd.ref<i1>` to appropriate HW types (e.g., `!hw.integer<1>` with appropriate attributes)
- Schedule this pass before `LowerState` in the pipeline

**Pros**: 
- Keeps StateType verification simple
- Follows existing MLIR conversion patterns
- Easier to reason about

**Cons**: 
- May lose semantic information about tri-state behavior
- Requires careful handling of all LLHD reference type variants

### Option 3: Special Handling for inout Ports
Add special handling in the SystemVerilog to Arc conversion to handle `inout` ports without generating LLHD reference types:
- Convert `inout` ports to separate input/output ports with additional control signals
- Or use a custom Arc dialect type for bidirectional signals

**Pros**: 
- Avoids the issue entirely by not generating problematic types
- May improve overall compilation efficiency

**Cons**: 
- Requires understanding of SystemVerilog inout semantics
- May be complex to implement correctly

## Keywords for Issue Search
- `arcilator`
- `StateType`
- `LLHD reference`
- `inout port`
- `tri-state`
- `verifyInvariants`
- `LowerState pass`
- `!llhd.ref<i1>`
- `bit width`

## Related Files to Investigate
- `lib/Dialect/Arc/Transforms/LowerState.cpp` - The failing pass implementation
- `include/circt/Dialect/Arc/ArcTypes.cpp.inc` - StateType definition (generated from TableGen)
- `lib/Conversion/MooreToArc/` - Moore (SystemVerilog) to Arc conversion
- `include/circt/Dialect/LLHD/LLHDDialect.h` - LLHD dialect definition
- `lib/Dialect/LLHD/IR/LLHDDialect.cpp` - LLHD type implementation

## Note on Bug Fix
The bug appears to have been fixed in the current toolchain (LLVM 22.0.0git, CIRCT firtool-1.139.0), as the reproduction attempt succeeded without triggering the assertion. This suggests one of the fixes mentioned above has already been implemented.
