# Root Cause Analysis Report

## Executive Summary

The CIRCT `Sig2RegPass` crashes with an assertion failure `"cannot RAUW a value with itself"` when processing a SystemVerilog module with a self-referential continuous assignment (`assign q_out = _02_ ? 1'b0 : q_out`). The crash occurs because the signal promotion pass attempts to replace a value with itself during the destruction of internal data structures, triggered by a circular dependency where the same signal is both read and written in a trivial assignment.

## Crash Context

- **Tool**: circt-verilog
- **Dialect**: LLHD
- **Failing Pass**: Sig2RegPass (Signal to Register Promotion)
- **Crash Type**: Assertion failure
- **CIRCT Version**: 1.139.0

## Error Analysis

### Assertion Message

```
Assertion `(!newValue || this != OperandType::getUseList(newValue)) && "cannot RAUW a value with itself"' failed.
```

This assertion in MLIR's `UseDefLists.h:213` prevents replacing all uses of a value with the same value, which would cause infinite loops.

### Key Stack Frames

```
#11 (anonymous namespace)::Offset::~Offset()           // Sig2RegPass.cpp:35
#12 (anonymous namespace)::Interval::~Interval()       // Sig2RegPass.cpp:54
#13 (anonymous namespace)::SigPromoter::promote()      // Sig2RegPass.cpp:286
#14 (anonymous namespace)::Sig2RegPass::runOnOperation() // Sig2RegPass.cpp:364
```

The crash occurs during the destruction of the `Interval` struct, specifically when its `Offset` member's `SmallVector<Value> dynamic` is being destroyed. The SmallVector destructor attempts to clean up Value references, triggering RAUW with self.

## Test Case Analysis

### Code Summary

The test creates a flip-flop module with conflicting signal drivers:
1. A sequential assignment in an `always @(negedge clock)` block
2. A combinational continuous assignment that creates a self-loop

### Key Constructs

- `always @(negedge clock)` - Edge-triggered sequential logic
- `assign q_out = _02_ ? 1'b0 : q_out` - Self-referential continuous assignment
- `_02_ = clock & ~clock` - Always evaluates to false (constant 0)

### Problematic Patterns

```systemverilog
// This creates a self-referential assignment
assign _02_ = _00_ & _01_;  // _02_ = clock & ~clock = always 0
assign q_out = _02_ ? 1'b0 : q_out;  // Effectively: q_out = q_out
```

The condition `_02_` is always false because `clock & ~clock` is always 0. This makes the assignment effectively `assign q_out = q_out`, a no-op that reads and writes the same signal.

Additionally, there are **multiple drivers** for `q_out`:
1. `always @(negedge clock) q_out <= d;` - Sequential driver
2. `assign q_out = ...` - Continuous driver

This is illegal in synthesizable SystemVerilog (multiple drivers to the same signal), but the crash occurs before this can be diagnosed.

## CIRCT Source Analysis

### Crash Location

**File**: `lib/Dialect/LLHD/Transforms/Sig2RegPass.cpp`
**Struct**: `Offset` (lines 35-50)

```cpp
struct Offset {
  Offset(uint64_t min, uint64_t max, ArrayRef<Value> dynamic)
      : min(min), max(max), dynamic(dynamic) {}
  // ...
  SmallVector<Value> dynamic;  // <-- Destruction triggers RAUW
};
```

### Code Context

The `promote()` function (lines 220-297) processes signal reads and writes:

```cpp
// Lines 267-289: Handle reads
for (auto interval : readIntervals) {
  // ... build read operations ...
  read = builder.createOrFold<hw::BitcastOp>(loc, interval.value.getType(), read);
  if (read != interval.value) {
    interval.value.replaceAllUsesWith(read);
  }
}
```

The check `if (read != interval.value)` protects the explicit `replaceAllUsesWith` call. However, the crash happens during destructor cleanup, not during the explicit RAUW call.

### Processing Path

1. Parse SystemVerilog input containing `assign q_out = _02_ ? 1'b0 : q_out`
2. Lower to LLHD dialect with signal operations
3. `Sig2RegPass` attempts to promote signals to registers
4. `SigPromoter::computeIntervals()` identifies read and write intervals
5. The self-referential assignment creates a read interval where `interval.value` references the same signal being written
6. `SigPromoter::promote()` processes intervals
7. When building the replacement value, `createOrFold` operations may produce the same value
8. During scope exit, `Interval` destructor runs, which destroys `Offset`
9. `SmallVector<Value> dynamic` destructor triggers cleanup
10. **CRASH**: The Value cleanup invokes RAUW where source == destination

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)

**Cause**: Self-referential continuous assignment (`assign q_out = ... q_out`) creates a circular dependency that confuses the Sig2Reg pass's interval tracking.

**Evidence**:
- Test case explicitly reads and writes the same signal in one continuous assignment
- The mux condition is constant-false, making this effectively `q_out = q_out`
- Stack trace shows crash during Interval/Offset destruction, indicating corrupted Value references
- `SmallVector<Value> dynamic` in Offset stores SSA values that become self-referential

**Mechanism**:
1. The `q_out` signal is used as both input (in the mux's else branch) and output
2. When Sig2Reg processes this, it creates an interval where the read value equals the signal being promoted
3. The `createOrFold` operations may fold to produce the original value
4. When the Interval struct is destroyed, the Value cleanup attempts to replace a value with itself

### Hypothesis 2 (Medium Confidence)

**Cause**: Multiple drivers to the same signal (`always_ff` and `assign`) creates conflicting intervals that aren't properly handled.

**Evidence**:
- `q_out` has two drivers: sequential (`always`) and combinational (`assign`)
- The `isPromotable()` check (lines 201-216) only checks for overlapping offset ranges, not multiple write sources
- The signal may have conflicting interval entries that share the same SSA value

**Mechanism**:
1. Both the `always` block and `assign` create write intervals for `q_out`
2. These intervals may share or conflict in their SSA value references
3. During promotion, one interval's RAUW affects another's values
4. The destructor cleanup encounters the corrupted state

### Hypothesis 3 (Lower Confidence)

**Cause**: Constant folding of `clock & ~clock = 0` causes unexpected behavior when the mux is eliminated.

**Evidence**:
- The condition `_02_` is always false
- CIRCT may constant-fold this to directly assign `q_out = q_out`
- The identity assignment may not be properly filtered out

## Suggested Fix Directions

1. **Add self-referential assignment detection**: Before processing intervals, check if a signal reads from itself in a continuous assignment and either emit an error or skip that signal.

2. **Guard against self-RAUW in interval processing**: In the `promote()` function, add checks not just for `read != interval.value` but also verify the underlying use-def chains don't form cycles.

3. **Detect multiple drivers early**: Add validation in `computeIntervals()` to detect and reject signals with multiple drivers from different source types (combinational vs sequential).

4. **Improve isPromotable() validation**: Add a check for signals that are used in their own definition (self-loops).

## Keywords for Issue Search

Sig2RegPass RAUW replaceAllUsesWith self-referential assign multiple-drivers signal-promotion LLHD

## Related Files

- `lib/Dialect/LLHD/Transforms/Sig2RegPass.cpp` - Crash location
- `llvm/mlir/include/mlir/IR/UseDefLists.h` - RAUW assertion
- `include/circt/Dialect/LLHD/IR/LLHDOps.td` - LLHD operation definitions
- `lib/Conversion/MooreToCore/` - Moore dialect lowering to Core dialects
