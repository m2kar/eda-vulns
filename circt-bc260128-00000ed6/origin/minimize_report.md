# Minimization Report

## Summary

| Metric | Value |
|--------|-------|
| Original file | source.sv (26 lines) |
| Minimized file | bug.sv (11 lines) |
| Reduction | **57.7%** |
| Crash preserved | N/A (Bug fixed in current version) |

## Root Cause Analysis

Based on `analysis.json`, the crash was triggered by:

- **Trigger Construct**: `inout wire` with tri-state assignment
- **Problematic Type**: `!llhd.ref<i1>` 
- **Failure Point**: `arc::StateType::get()` requires known bit width
- **Error Message**: `state type must have a known bit width; got '!llhd.ref<i1>'`

## Key Constructs Preserved

The minimized test case preserves only the essential constructs:

1. **`inout wire` port** - Creates the LLHD reference type
2. **Tri-state assignment** (`assign c = a ? 1'bz : 1'b0`) - Triggers ref type usage

## Removed Elements

| Element | Lines Removed | Reason |
|---------|---------------|--------|
| `output logic b` port | 1 | Not involved in crash path |
| `logic [63:0] clkin_data` | 1 | Not used in minimal repro |
| Sequential `always_ff` for `b` | 4 | Output not needed |
| Sequential `always_ff` for `clkin_data` | 3 | Counter not needed |
| Comments | 3 | Documentation only |
| Empty lines | 3 | Formatting only |

## Minimized Test Case

```systemverilog
// Minimized test case for CIRCT arc::StateType assertion failure
// Original: 260128-00000ed6
// Trigger: inout wire with tri-state assignment creating !llhd.ref<i1>
module MixPorts(
  input  logic clk,
  input  logic a,
  inout  wire  c
);
  // Tri-state assignment to inout wire - creates !llhd.ref<i1> type
  assign c = a ? 1'bz : 1'b0;
endmodule
```

## Verification

### Current Toolchain Result

```
Tool: circt-verilog + arcilator (firtool-1.139.0)
Status: SUCCESS (Bug is FIXED)
Exit Code: 0
```

The pipeline now successfully compiles without assertion failure, producing valid LLVM IR.

### Original Error (Historical)

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
Assertion `succeeded(ConcreteT::verifyInvariants(...))' failed.
```

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv | arcilator
```

## Conclusion

The test case has been minimized from 26 lines to 11 lines (57.7% reduction) while preserving the exact construct (`inout wire` with tri-state assignment) that originally triggered the `arc::StateType` assertion failure. The bug has been **fixed** in the current CIRCT version (firtool-1.139.0).
