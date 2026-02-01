# Root Cause Analysis: LLHD Sig2Reg Pass Infinite Loop

## Executive Summary

**Bug Type**: Infinite loop / timeout in LLHD Sig2Reg pass
**Original Crash**: Assertion failure "cannot RAUW a value with itself"
**Current Behavior**: Compiler hangs indefinitely
**Root Cause**: Combinational loop in test case causes circular dependency during signal-to-register promotion
**Severity**: High (compiler hang, no error message)

## Error Context

### Original Assertion (from fuzzer)
```
Assertion `(!newValue || this != OperandType::getUseList(newValue)) && "cannot RAUW a value with itself"' failed.
```

**Location**: `mlir/IR/UseDefLists.h:213` in `replaceAllUsesWith()`

### Stack Trace Analysis

Key frames from original crash:
```
#13 (anonymous namespace)::Offset::~Offset() Sig2RegPass.cpp:35
#14 (anonymous namespace)::Interval::~Interval() Sig2RegPass.cpp:54
#15 (anonymous namespace)::SigPromoter::promote() Sig2RegPass.cpp:286
#16 (anonymous namespace)::Sig2RegPass::runOnOperation() Sig2RegPass.cpp:364
```

The crash occurs during cleanup in `SigPromoter::promote()` when attempting to replace uses of a value with itself.

## Test Case Analysis

### Problematic Code Pattern

```systemverilog
module example_module(input logic clock, output logic q_out);
  wire _00_, _01_, _02_;
  logic d;

  assign _00_ = clock;
  assign _01_ = ~clock;
  assign _02_ = _00_ & _01_;  // Always 0 (clock & ~clock)

  always @(negedge clock) begin
    q_out <= d;  // Non-blocking assignment
  end

  assign q_out = _02_ ? 1'b0 : q_out;  // ⚠️ COMBINATIONAL LOOP
endmodule
```

### Key Constructs

1. **Negedge sensitivity**: `always @(negedge clock)`
2. **Non-blocking assignment**: `q_out <= d`
3. **Continuous assignment to same signal**: `assign q_out = ... q_out`
4. **Combinational loop**: `q_out` depends on itself

### Why This is Invalid

The code violates IEEE 1800 semantics:
- `q_out` is driven by both procedural (always block) and continuous assignment
- The continuous assignment creates a combinational loop: `q_out = q_out`
- This should be rejected during elaboration/lowering

## Root Cause Hypothesis

### Primary Hypothesis (High Confidence)

**Combinational loop causes infinite recursion in signal promotion**

**Evidence**:
1. `q_out` has two drivers:
   - Procedural: `always @(negedge clock) q_out <= d`
   - Continuous: `assign q_out = _02_ ? 1'b0 : q_out`

2. During LLHD lowering, the Sig2Reg pass attempts to promote signals to registers

3. When processing the continuous assignment, it encounters a self-reference:
   - Old value: `q_out` (from always block)
   - New value: `_02_ ? 1'b0 : q_out` (depends on old `q_out`)

4. The RAUW (Replace All Uses With) operation detects this circular dependency and asserts

5. In the current version, the assertion was likely removed or the code path changed, causing an infinite loop instead

### Secondary Hypothesis (Medium Confidence)

**Missing combinational loop detection in LLHD lowering**

The compiler should detect and reject combinational loops during:
- Moore-to-Core lowering (SystemVerilog → HW dialect)
- HW-to-LLHD lowering
- LLHD Sig2Reg pass

The fact that it reaches Sig2Reg without error indicates missing validation.

## CIRCT Source Analysis

### Relevant Files

1. **lib/Dialect/LLHD/Transforms/Sig2RegPass.cpp**
   - Line 286: `SigPromoter::promote()` - where crash occurs
   - Line 364: `Sig2RegPass::runOnOperation()` - entry point

2. **mlir/IR/UseDefLists.h:213**
   - `replaceAllUsesWith()` - RAUW implementation with self-reference check

### Expected Behavior

The compiler should:
1. Detect the combinational loop during lowering
2. Emit a diagnostic error like:
   ```
   error: combinational loop detected: signal 'q_out' depends on itself
   ```
3. Fail compilation gracefully (not hang or assert)

## Reproduction Status

### Original Crash (CIRCT 1.139.0)
- **Command**: `circt-verilog --ir-hw test.sv | arcilator | opt -O0 | llc -O0`
- **Result**: Assertion failure in RAUW

### Current Behavior (firtool-1.139.0)
- **Command**: `circt-verilog --ir-hw test.sv`
- **Result**: Infinite loop / timeout (no output)

**Conclusion**: Bug behavior changed but still exists. The self-reference detection may have been removed or bypassed, causing infinite recursion instead of assertion.

## Impact Assessment

**Severity**: High
- Compiler hangs indefinitely (DoS)
- No error message to guide user
- Affects any code with combinational loops

**Affected Components**:
- LLHD Sig2Reg pass
- Possibly Moore-to-Core lowering (missing validation)

## Recommended Fix

### Short-term Fix
Add combinational loop detection in `Sig2RegPass::runOnOperation()`:
1. Build dependency graph of signal assignments
2. Detect cycles using DFS/topological sort
3. Emit error and fail gracefully

### Long-term Fix
Add validation earlier in the pipeline:
1. Moore-to-Core: Detect multiple drivers to same signal
2. HW-to-LLHD: Validate signal dependency graph
3. Add IEEE 1800 compliance checks for procedural/continuous assignment conflicts

## Keywords for Duplicate Search

- LLHD Sig2Reg
- combinational loop
- RAUW self-reference
- signal promotion
- infinite loop
- timeout
- negedge
- circular dependency
- multiple drivers
