# [circt-verilog][LLHD] Compiler hangs on invalid code with procedural and continuous assignment to same signal

## Description

CIRCT hangs indefinitely when processing invalid SystemVerilog code that has both procedural (non-blocking) and continuous assignments to the same signal. Other tools (Verilator, Icarus Verilog) correctly reject this code with an error message. CIRCT should detect this invalid pattern and report an error instead of hanging.

## Steps to Reproduce

1. Save the following code as `bug.sv`
2. Run: `circt-verilog --ir-hw bug.sv`
3. Observe: Compiler hangs indefinitely (no output, no error message)

## Test Case

```systemverilog
module bug(input logic clk, output logic q);
  logic d;

  always @(negedge clk) begin
    q <= d;  // Non-blocking procedural assignment
  end

  assign q = q;  // Continuous assignment to same signal
endmodule
```

## Expected Behavior

CIRCT should reject this code with an error message similar to:
```
error: Cannot perform procedural assignment to variable 'q' because it is also continuously assigned
```

## Actual Behavior

Compiler hangs indefinitely in the LLHD Sig2Reg pass (no output after 60+ seconds).

## Root Cause Analysis

The test case violates IEEE 1800 semantics by having multiple drivers to signal `q`:
- Procedural driver: `always @(negedge clk) q <= d`
- Continuous driver: `assign q = q`

The continuous assignment also creates a combinational loop (`q` depends on itself).

During LLHD lowering, the Sig2Reg pass attempts to promote signals to registers but encounters a circular dependency when processing the self-referencing assignment. This causes infinite recursion in the signal promotion logic.

## Cross-Tool Validation

- **Icarus Verilog**: ✅ Correctly rejects
  ```
  error: Cannot perform procedural assignment to variable 'q' because it is also continuously assigned.
  ```

- **Verilator**: ✅ Correctly rejects
  ```
  %Error-BLKANDNBLK: Blocked and non-blocking assignments to same variable: 'q'
  ```

- **CIRCT**: ❌ Hangs indefinitely

## Suggested Fix

Add validation in the LLHD lowering pipeline to detect:
1. Multiple drivers to the same signal (procedural + continuous)
2. Combinational loops in signal assignments

The compiler should emit a diagnostic error and fail gracefully instead of hanging.

## Environment

- **CIRCT Version**: firtool-1.139.0
- **LLVM Version**: 22.0.0git
- **Affected Component**: LLHD Sig2Reg pass
