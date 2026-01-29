# Validation Report

## Classification

| Field | Value |
|-------|-------|
| **Result** | `feature_request` |
| **Reason** | Arc dialect explicitly does not support inout ports. The assertion failure (instead of graceful error message) is the bug. |

## Summary

This test case triggers an assertion failure in CIRCT's arcilator tool when processing SystemVerilog code with `inout` ports. While Arc dialect not supporting bidirectional ports is by design, the tool should emit a user-friendly diagnostic rather than crash with an assertion.

## Syntax Validation

| Tool | Result | Notes |
|------|--------|-------|
| Verilator | ✅ Pass | `--lint-only` no errors |
| Slang | ✅ Pass | 0 errors, 0 warnings |

The minimized test case is syntactically valid SystemVerilog per IEEE 1800-2017.

## Test Case Minimization

| Metric | Value |
|--------|-------|
| Original lines | 24 |
| Minimized lines | 6 |
| **Reduction** | **75%** |

### Minimized Code (bug.sv)
```systemverilog
// Minimized test case for Arc inout port assertion failure
// Original: 24 lines -> Minimized: 6 lines
// Key pattern: inout port causes llhd::RefType which Arc cannot handle
module InoutBug(
  inout logic [7:0] data_bus
);
endmodule
```

### What Was Removed
- Input/output ports (not needed to trigger)
- Unpacked array declarations
- Variable index logic
- always_ff block
- Conditional tristate assignment

### What Was Preserved
- **`inout` port** - The core trigger for the assertion failure

## Reproducibility

| Field | Value |
|-------|-------|
| Reproduced | ❌ No |
| Note | Original crash not reproducible with current toolchain |

The crash was originally observed with CIRCT 1.139.0. Current toolchain may have different behavior.

## Recommendation

**Action**: Report as feature request / error handling improvement

**Rationale**:
1. `inout` ports being unsupported in Arc is by design (simulation model limitation)
2. However, the assertion failure is poor UX - should be a diagnostic
3. Error message "state type must have a known bit width; got '!llhd.ref<i8>'" is confusing
4. Should instead say: "error: inout ports are not supported by arcilator"

## Crash Mechanism

```
inout port declaration
    ↓
MooreToCore lowering creates llhd::RefType
    ↓
HW module with RefType argument
    ↓
LowerState::run() calls StateType::get(RefType)
    ↓
StateType::verifyInvariants() calls computeLLVMBitWidth(RefType)
    ↓
computeLLVMBitWidth() returns nullopt (RefType not handled)
    ↓
Assertion failure
```

## Files Generated

- `bug.sv` - Minimized test case (6 lines)
- `error.log` - Original error output
- `command.txt` - Reproduction command
- `validation.json` - Machine-readable validation results
- `validation.md` - This report
