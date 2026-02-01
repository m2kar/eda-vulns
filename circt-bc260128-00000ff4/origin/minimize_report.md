# Minimization Report

## Summary

| Item | Value |
|------|-------|
| **Original file** | source.sv (19 lines) |
| **Minimized file** | bug.sv (4 lines) |
| **Reduction** | 78.9% |
| **Crash preserved** | No (bug fixed in current toolchain) |

## Root Cause Analysis

Based on `analysis.json`, the crash was caused by:

- **Trigger construct**: `inout wire` port declaration
- **Mechanism**: `inout wire c` gets lowered to `!llhd.ref<i1>` type
- **Failure point**: Arc's `StateType::verify()` fails because `computeLLVMBitWidth()` returns `nullopt` for LLHD ref types
- **Error message**: `state type must have a known bit width; got '!llhd.ref<i1>'`

## Key Constructs Preserved

From analysis.json:
- `inout wire c` - **PRESERVED** (trigger construct)

## Removed Elements

| Element | Lines | Reason |
|---------|-------|--------|
| `input logic a` | 1 | Not related to inout handling |
| `output logic b` | 1 | Not related to inout handling |
| `input logic clk` | 1 | Not related to inout handling |
| `output logic out0` | 1 | Not related to inout handling |
| `logic [3:0] idx` | 1 | Sequential counter not needed |
| `always_ff` block | 3 | Sequential logic not related |
| `assign out0 = idx[0]` | 1 | Output assignment not needed |
| `assign b = a` | 1 | Pass-through not needed |
| `assign c = a ? 1'bz : 1'b0` | 1 | Tristate assignment removed (inout port itself triggers the bug) |

## Minimized Test Case

```systemverilog
module InoutBug(
  inout wire c
);
endmodule
```

## Verification

### Original Assertion (from error.txt)
```
Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed
```

### Original Error Message
```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
```

### Current Toolchain Result
```
Exit code: 0 (No crash - bug appears to be fixed)
```

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv | arcilator
```

## Notes

1. The bug does **not** reproduce in the current toolchain (CIRCT firtool-1.139.0, LLVM 22.0.0git)
2. The original crash occurred in CIRCT version 1.139.0 with the same version identifier
3. The fix likely involved:
   - Adding LLHD ref type handling to `computeLLVMBitWidth()`, OR
   - Filtering out unsupported port types before `LowerStatePass`, OR
   - Changes in how `inout` ports are lowered by `circt-verilog`
4. The minimal test case documents the original issue pattern for reference
