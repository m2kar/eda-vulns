# Root Cause Analysis Report

## Executive Summary

Bug triggered by arcilator's LowerState pass attempting to create a StateType for an `!llhd.ref<i1>` type (LLHD reference type) that results from lowering inout ports in SystemVerilog modules. The pass assumes all module arguments are bit-width types but fails to handle LLHD reference types, causing an assertion failure: "state type must have a known bit width".

## Crash Context

- **Tool/Command**: arcilator (after circt-verilog --ir-hw)
- **Dialect**: Arc dialect (circt::arc)
- **Failing Pass**: LowerState pass
- **Crash Type**: Assertion failure
- **CIRCT Version**: 1.139.0 (original issue), 22.0.0git (current - bug appears fixed)

## Error Analysis

### Assertion/Error Message

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: .../LowerState.cpp:219:66: Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Key Stack Frames

```
#12 circt::arc::StateType::get(mlir::Type) at ArcTypes.cpp.inc:108
#13 (anonymous namespace)::ModuleLowering::run() at LowerState.cpp:219:66
#14 (anonymous namespace)::LowerStatePass::runOnOperation() at LowerState.cpp:1198:41
```

The crash occurs in `ModuleLowering::run()` when it tries to allocate state for module inputs.

## Test Case Analysis

### Code Summary

A simple SystemVerilog module with:
- 3 input ports: `clk`, `a`, `inout c`
- 1 output port: `b`
- Sequential logic with `always @(posedge clk)`
- For loop for register assignment
- Inout port assigned from internal register

### Key Constructs

- **inout port**: Port `c` is declared as `inout logic c`
- **Sequential logic**: `always @(posedge clk)` block with register updates
- **For loop**: Loop writing to each bit of `temp_reg`

### Potentially Problematic Patterns

1. **inout port in sequential context**: The inout port is assigned from `temp_reg[1]`, which is updated in a sequential block
2. **Assignment from register to inout**: `assign c = temp_reg[1];` connects a sequential element to an inout port

### Intermediate MLIR Representation

After `circt-verilog --ir-hw`, the module is lowered to:

```mlir
module {
  hw.module @MixedPorts(in %clk : i1, in %a : i1, out b : i1, in %c : !llhd.ref<i1>) {
    // ... body ...
  }
}
```

Notice that the inout port `c` is represented as `in %c : !llhd.ref<i1>`, an LLHD reference type.

## CIRCT Source Analysis

### Crash Location

**File**: `lib/Dialect/Arc/Transforms/LowerState.cpp`
**Function**: `ModuleLowering::run()`
**Line**: 219 (original CIRCT 1.139.0)

### Code Context

```cpp
// Lines 214-221 in LowerState.cpp
// Allocate storage for inputs.
for (auto arg : moduleOp.getBodyBlock()->getArguments()) {
  auto name = moduleOp.getArgName(arg.getArgNumber());
  auto state =
      RootInputOp::create(allocBuilder, arg.getLoc(),
                          StateType::get(arg.getType()), name, storageArg);  // <-- Line 219
  allocatedInputs.push_back(state);
}
```

The code iterates through all module arguments and tries to create a `StateType` for each argument's type.

### Processing Path

1. **circt-verilog** lowers SystemVerilog to HW dialect
   - Inout ports are converted to LLHD reference types (`!llhd.ref<T>`)
   - This is the expected behavior for representing bidirectional ports

2. **arcilator** runs LowerState pass
   - Pass iterates over all module arguments
   - Calls `StateType::get(arg.getType())` for each argument
   - For `!llhd.ref<i1>`, `StateType::get()` calls verification invariants
   - Verification fails: LLHD ref types don't have a known bit width
   - Assertion fires

3. **Root Issue**: `StateType::get()` expects bit-width types (like `i1`, `i4`, etc.), but receives a reference type

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)

**Cause**: LowerState pass does not handle LLHD reference types that result from inout ports. When a module contains an inout port, circt-verilog converts it to an `!llhd.ref<T>` type. LowerState pass assumes all module arguments are bit-width types and attempts to create StateType objects for them, but StateType cannot represent reference types, causing the assertion failure.

**Evidence**:
- Stack trace shows crash at `StateType::get(arg.getType())` where `arg.getType()` is `!llhd.ref<i1>`
- Test case has an inout port `c` that is converted to `!llhd.ref<i1>`
- Intermediate MLIR confirms the port type is `!llhd.ref<i1>`
- Error message explicitly states: "state type must have a known bit width; got '!llhd.ref<i1>'"
- Related GitHub issue #9574 title: "[Arc] Assertion failure when lowering inout ports in sequential logic"

**Mechanism**:
1. User writes SystemVerilog with inout port
2. circt-verilog lowers to HW dialect with LLHD ref type for the port
3. arcilator's LowerState pass iterates over all module arguments
4. Pass assumes all arguments are bit-width types suitable for state storage
5. `StateType::get(!llhd.ref<i1>)` fails verification: reference types don't have bit width
6. Assertion failure triggers abort

### Hypothesis 2 (Low Confidence)

**Cause**: The interaction between circt-verilog and arcilator regarding inout ports is not well-defined. There may be a missing conversion step between the two tools that should transform LLHD ref types into something Arc dialect can handle.

**Evidence**:
- The pipeline uses `circt-verilog --ir-hw | arcilator`
- No explicit conversion between LLHD and Arc dialects in the command
- Both dialects handle different aspects (LLHD for simulation, Arc for state machine modeling

**Counter-evidence**:
- This would affect all modules with inout ports, not just those with sequential logic
- Related issue specifically mentions "in sequential logic", suggesting the interaction with state storage

## Suggested Fix Directions

1. **Filter out LLHD ref types in LowerState**:
   - Skip creating state storage for arguments with LLHD ref types
   - Inout ports are typically handled differently than regular state
   - May need special handling in the Arc lowering pipeline

2. **Convert LLHD refs before Arc lowering**:
   - Add a conversion pass that transforms `!llhd.ref<T>` to a type Arc can handle
   - This would be inserted between circt-verilog and arcilator
   - Requires defining how inout semantics should be represented in Arc

3. **Add validation before state allocation**:
   - Check if argument type is compatible with StateType before creating it
   - Provide a better error message if incompatible type is found
   - Don't crash with assertion; report a proper diagnostic

## Keywords for Issue Search

- `inout port arcilator`
- `LLHD ref type LowerState`
- `StateType known bit width`
- `arc dialect inout`
- `sequential logic inout`

## Related Files to Investigate

- `lib/Dialect/Arc/Transforms/LowerState.cpp` - Contains the crash site, handles state lowering
- `lib/Dialect/Moore/Conversions/VerilogToHW.cpp` - Converts Verilog/SV to HW dialect, may generate LLHD refs
- `include/circt/Dialect/Arc/ArcOps.td` - Defines Arc types and operations
- `lib/Dialect/LLHD/IR/LLHDDialect.cpp` - Defines LLHD reference type semantics

## Related GitHub Issues

- Issue #9574: "[Arc] Assertion failure when lowering inout ports in sequential logic" (Created 2026-02-01, OPEN)
- This issue appears to describe the same problem
- Both involve inout ports in sequential context triggering arcilator assertion failures

## Reproduction Status

- **Original Version (1.139.0)**: Bug reproduces as described
- **Current Version (22.0.0git)**: Bug does NOT reproduce - appears to be fixed
- **Related Issue**: Issue #9574 is still open, suggesting the fix may not be complete or merged

## Conclusion

The root cause is a type mismatch in LowerState pass: it attempts to create StateType for LLHD reference types (from inout ports) but StateType requires bit-width types. The fix should either skip LLHD ref types during state allocation or convert them to a compatible type before Arc lowering. The issue appears to be addressed in CIRCT 22.0.0git but may still exist in the 1.139.0 version where it was discovered.
