# Validation Report: circt-b6

## Classification: REPORT

**Confidence**: High

## Summary

This is a legitimate compiler bug. CIRCT crashes with an assertion failure when compiling syntactically valid SystemVerilog code that uses inout ports within always_ff blocks.

## Syntax Validation

**SystemVerilog Validity**: ✓ VALID

The minimized test case uses standard IEEE 1800-2017 SystemVerilog constructs:
- `inout wire` declarations (bidirectional ports)
- `always_ff` sequential blocks
- Non-blocking assignments (`<=`)
- Clock edge sensitivity (`@(posedge clk)`)

All syntax is compliant with the SystemVerilog standard.

## Bug Characteristics

- **Type**: Compiler crash (assertion failure)
- **Severity**: High
- **Affected Tool**: arcilator
- **Affected Pass**: LowerStatePass
- **Reproducibility**: 100%

## Root Cause

The Arc dialect's `LowerStatePass` attempts to create a `StateType` for sequential logic that reads from an inout port. The inout port is represented as `!llhd.ref<i1>` (LLHD reference type), which lacks a known bit width. The `StateType::verifyInvariants()` function rejects this type, causing an assertion failure instead of emitting a proper diagnostic.

**Error Message**:
```
Assertion `succeeded(result.verifyInvariants(loc))' failed.
state type must have a known bit width; got '!llhd.ref<i1>'
```

## Expected vs Actual Behavior

**Expected**: Either successful compilation or a graceful error message like:
```
error: inout ports cannot be used directly in sequential logic
```

**Actual**: Compiler crash with assertion failure and stack dump.

## Minimization Quality

- **Original**: 21 lines
- **Minimized**: 9 lines
- **Reduction**: 57.14%

The minimized test case removes all irrelevant constructs while preserving the crash:
- Removed unused input/output ports
- Removed unnecessary assignments
- Simplified data types to minimum required
- Kept only the problematic construct: inout port in always_ff

## Cross-Tool Validation

Cross-tool validation was not performed due to tool availability constraints. However, the SystemVerilog syntax is standard-compliant and should be accepted by any IEEE 1800-2017 compliant compiler.

## Recommendation

**Action**: Report to CIRCT issue tracker

**Priority**: High - This affects users compiling valid SystemVerilog designs with bidirectional ports in sequential logic.

**Suggested Labels**: `bug`, `Arc`, `LLHD`, `type-system`

**Suggested Title**: `[Arc] Assertion failure when lowering inout ports in sequential logic`

## Technical Details

**Crash Location**: `lib/Dialect/Arc/Transforms/LowerState.cpp:219`

**Type Flow**:
1. Frontend parses inout port → `!llhd.ref<i1>`
2. Arc LowerStatePass attempts `StateType::get(!llhd.ref<i1>)`
3. StateType verification fails → assertion

**Suggested Fix**: Add type checking before `StateType::get()` to either:
1. Dereference LLHD reference types, or
2. Emit a proper diagnostic error message

## Conclusion

This is a valid bug report. The test case is minimal, reproducible, and demonstrates a compiler crash on valid input. It should be reported to the CIRCT project.
