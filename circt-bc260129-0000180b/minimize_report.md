# Minimization Report

## Summary
- **Original file**: origin/source.sv (37 lines)
- **Minimized file**: bug.sv (17 lines)
- **Reduction**: 54.1%
- **Error preserved**: Yes

## Preservation Analysis

### Key Constructs Preserved
Based on `analysis.json`, following constructs were kept:
- **Self-referential typedef**: `typedef container#(pkg::my_class) my_type`
- **Parameterized class**: `container#(type T = int)`
- **Package**: `package pkg ... endpackage`
- **Class instantiation**: `class_obj = new()`

### Removed Elements
- Sub module (4 lines) - Not required to trigger error
- Wire declarations (3 lines) - logic_sig, result_sig, test_signal
- Module instantiation (1 line) - Sub module instantiation
- Assignment statements (2 lines) - Using test_signal
- Display statement (1 line) - $display call
- Comments (6 lines) - All comments
- Extra blank lines (3 lines)

## Critical Finding

### The Role of Delay Statement
The delay statement `#1` is **CRITICAL** for reproducing the error:
- **With delay (#1)**: Triggers error with `hw.bitcast` and overflow bitwidth `i1073741823`
- **Without delay**: Compiles successfully, creates correct LLHD process

This indicates the error occurs in a specific code path triggered when:
1. LLHD process needs to handle time delays
2. The HoistSignals pass attempts to materialize drive values with delays
3. The self-referential type's bit width cannot be calculated
4. This leads to either assertion failure (original) or overflow error (current)

## Verification

### Original Error
```
origin/source.sv:28:3: error: 'hw.bitcast' op result #0 must be Type wherein the bitwidth in hardware is known, but got '!llvm.ptr'
  initial begin
  ^
origin/source.sv:28:3: note: see current operation: %20 = "hw.bitcast"(%19) : (i1073741823) -> !llvm.ptr
```

### Final Error
```
bug.sv:16:3: error: 'hw.bitcast' op result #0 must be Type wherein the bitwidth in hardware is known, but got '!llvm.ptr'
  initial begin
  ^
bug.sv:16:3: note: see current operation: %6 = "hw.bitcast"(%5) : (i1073741823) -> !llvm.ptr
```

**Match**: ✅ Exact match (same error message, same bitwidth value)

## Reproduction Command

```bash
export PATH=/opt/llvm-22/bin:$PATH && export PATH=/opt/firtool/bin:$PATH && circt-verilog --ir-hw bug.sv 2>&1
```

## Minimization History

| Iteration | Lines | Removed | Result |
|-----------|--------|----------|---------|
| minimal1.sv | 8 | Package only | No error (no module) |
| minimal2.sv | 18 | + module | No error (no delay) |
| minimal3.sv | 26 | + Sub, signals | No error (no delay) |
| minimal4.sv | 31 | + display | ✅ Error reproduced |
| minimal5.sv | 31 | = original | ✅ Error reproduced |
| minimal6.sv | 26 | - Sub module | ✅ Error reproduced |
| minimal7.sv | 22 | - display | ✅ Error reproduced |
| minimal8.sv | 18 | - delay | No error |
| bug.sv | 17 | - logic signals | ✅ Error reproduced |

## Insights

### Why Delay Matters
The delay `#1` triggers the LLHD HoistSignals pass to:
1. Materialize drive values with time delays
2. Call `hw::getBitWidth()` to create default constants
3. Attempt to calculate bit width of self-referential type
4. Either return `-1` (causing assertion) or overflow (causing i1073741823)

Without delays, the process doesn't need materialization, bypassing the problematic code path.

### Bitwidth Value
The overflow value `1073741823` = 2^30 - 1 suggests:
1. Unbounded recursion in type size calculation
2. Signed 32-bit integer overflow somewhere in the path
3. The compiler attempts to accumulate type sizes without cycle detection

## Notes

- The Sub module is completely unnecessary for reproducing the error
- Wire signals and assignments are not needed
- The display statement is not needed (delay alone is sufficient)
- The core issue is: self-referential typedef + delay statement

## Recommendation

When filing the issue, emphasize that:
1. The error specifically requires a delay statement in the initial block
2. Without delays, the self-referential typedef compiles successfully
3. This suggests the bug is in HoistSignals pass handling of timed drives
4. The overflow bitwidth `i1073741823` indicates unbounded recursion
