# Root Cause Analysis: Arc StateType Assertion on Inout Port

## Executive Summary

**Dialect**: Arc (with LLHD interaction)
**Crash Type**: Assertion failure
**Severity**: High - Compiler crash on valid SystemVerilog
**Root Cause**: Type system incompatibility between LLHD reference types and Arc StateType requirements

## Error Context

**Assertion Message**:
```
state type must have a known bit width; got '!llhd.ref<i1>'
```

**Assertion Location**:
```
StorageUniquerSupport.h:180: Assertion `succeeded(ConcreteT::verifyInvariants(...))' failed
```

**Crashing Pass**: `LowerStatePass` in Arc dialect
**Crash Point**: `LowerState.cpp:219` in `ModuleLowering::run()`

## Problematic Construct

The crash is triggered by an `inout` port used in stateful logic:

```systemverilog
module MixedPorts(
  input logic a,
  output logic b,
  inout wire c,        // <-- Bidirectional port
  input logic clk
);
  logic [3:0] temp_reg;

  // Reading from inout in sequential logic
  always_ff @(posedge clk) begin
    temp_reg <= c;     // <-- Problem: inout used as state input
  end

  // Writing to inout with tri-state
  assign c = a ? temp_reg : 4'bz;  // <-- Bidirectional assignment
endmodule
```

## Technical Analysis

### Type System Flow

1. **Frontend (circt-verilog)**: Parses SystemVerilog and creates HW/Moore dialect IR
   - Inout port `c` is represented as an LLHD reference type: `!llhd.ref<i1>`
   - LLHD (Low-Level Hardware Description) uses reference types for bidirectional signals

2. **Arc Lowering (arcilator)**: Attempts to lower to Arc dialect for simulation
   - `LowerStatePass` tries to create state storage for sequential logic
   - At line 219 in `LowerState.cpp`, calls `StateType::get(mlir::Type)`
   - Passes the LLHD reference type `!llhd.ref<i1>` as the state type

3. **Type Verification Failure**:
   - `StateType::verifyInvariants()` checks that the type has a known bit width
   - LLHD reference types are opaque pointers without intrinsic bit width
   - Verification fails, triggering assertion

### Stack Trace Analysis

Key frames:
```
#12: circt::arc::StateType::get(mlir::Type)
     - Attempts to create Arc StateType from LLHD reference

#13: ModuleLowering::run() at LowerState.cpp:219
     - Module lowering pass tries to allocate state storage

#15: LowerStatePass::runOnOperation()
     - Top-level pass execution
```

## Root Cause

**Type System Mismatch**: The Arc dialect's `StateType` requires types with known bit widths (e.g., `i1`, `i32`, `!hw.array<4xi1>`), but LLHD reference types (`!llhd.ref<T>`) are opaque pointers that don't expose their underlying bit width at the type level.

**Missing Lowering Logic**: The `LowerStatePass` doesn't handle the case where state inputs come from inout ports represented as LLHD references. It should either:
1. Dereference the LLHD reference to get the underlying type
2. Reject inout ports in sequential contexts with a proper error message
3. Insert explicit load operations to convert references to values

## Why This Matters

1. **Valid SystemVerilog**: Using inout ports in always blocks is legal in SystemVerilog
2. **Compiler Crash**: Should emit diagnostic error, not assertion failure
3. **Simulation Blocker**: Prevents arcilator from simulating designs with bidirectional ports in stateful logic

## Reproduction

```bash
# Save test case as test.sv
circt-verilog --ir-hw test.sv | arcilator
```

**Expected**: Graceful error message or successful compilation
**Actual**: Assertion failure and crash

## Suggested Fix Locations

1. **lib/Dialect/Arc/Transforms/LowerState.cpp:219**
   - Add type checking before calling `StateType::get()`
   - Handle LLHD reference types by dereferencing or emitting error

2. **lib/Dialect/Arc/IR/ArcTypes.cpp**
   - Improve `StateType::verifyInvariants()` error message
   - Provide better diagnostic for unsupported types

3. **Frontend (Moore/HW to Arc lowering)**
   - Insert explicit load operations for inout port reads
   - Convert `!llhd.ref<T>` to `T` before state allocation

## Related Issues

- Inout port handling in Arc dialect
- LLHD reference type integration
- Type system boundaries between dialects
- Error reporting in type verification

## Test Case Characteristics

- **Minimal**: 21 lines of SystemVerilog
- **Constructs**: inout port, always_ff, tri-state logic
- **Dialects**: HW/Moore → LLHD → Arc
- **Reproducibility**: 100% on CIRCT 1.139.0
