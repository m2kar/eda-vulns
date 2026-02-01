# Minimization Report

## Summary

| Metric | Value |
|--------|-------|
| **Original file** | source.sv (19 lines) |
| **Minimized file** | bug.sv (2 lines) |
| **Reduction** | 89.5% |
| **Crash preserved** | N/A (historical bug, not reproducible) |

## Minimization Strategy

Since this is a **historical bug** (not reproducible in current toolchain), the minimization was performed based on:

1. **Root Cause Analysis** (`analysis.json`)
   - Identified `inout logic c` as the trigger construct
   - The crash occurs when arcilator's LowerState pass encounters `!llhd.ref<i1>` type

2. **Code Analysis**
   - The original `source.sv` contained additional constructs (clk, inputs, outputs, logic declarations, always_ff block)
   - Only the `inout logic c` port was required to trigger the bug

## Key Constructs Preserved

Based on `analysis.json`, the following construct was kept:

| Construct | Reason |
|-----------|--------|
| `inout logic c` | Root cause trigger - creates `!llhd.ref<i1>` type that LowerState cannot handle |

## Removed Elements

| Element | Lines Removed | Reason |
|---------|---------------|--------|
| `input logic clk` | 1 | Not related to inout port handling |
| `input logic signed [15:0] a` | 1 | Not related to inout port handling |
| `input logic [15:0] b` | 1 | Not related to inout port handling |
| `output logic out_b` | 1 | Not related to inout port handling |
| `logic signed [15:0] signed_result` | 1 | Internal signal, not needed |
| `assign signed_result = ...` | 1 | Continuous assignment, not needed |
| `always_ff @(posedge clk)` block | 3 | Sequential logic, not needed |
| Comments | 2 | Not functional |
| Empty lines | 4 | Not functional |

## Minimized Test Case

```systemverilog
module Test(inout logic c);
endmodule
```

## Original vs Minimized

### Original (`source.sv`)
```systemverilog
module MixedPorts(
  input  logic        clk,
  input  logic signed [15:0] a,
  input  logic        [15:0] b,
  output logic        out_b,
  inout  logic        c
);

  logic signed [15:0] signed_result;

  // Continuous assignment using both signed and unsigned ports
  assign signed_result = a + $signed(b);

  // Sequential logic with clock edge
  always_ff @(posedge clk) begin
    out_b <= signed_result[0];
  end

endmodule
```

### Minimized (`bug.sv`)
```systemverilog
module Test(inout logic c);
endmodule
```

## Verification

### Original Assertion (from error.txt)
```
state type must have a known bit width; got '!llhd.ref<i1>'
Assertion `succeeded(ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Current Status
- **Reproducible**: No (bug fixed in current toolchain)
- **Classification**: historical_bug

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv | arcilator
```

**Note**: This command was valid for CIRCT 1.139.0. The bug has been fixed in later versions.

## Notes

- Module name changed from `MixedPorts` to `Test` for simplicity
- All non-essential ports and logic removed
- Single `inout logic c` port is sufficient to demonstrate the bug
- The `inout` port triggers the creation of `!llhd.ref<i1>` type in MLIR
- This type was not handled by `computeLLVMBitWidth()` in LowerState pass
