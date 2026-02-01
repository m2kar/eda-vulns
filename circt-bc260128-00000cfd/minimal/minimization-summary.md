# Minimization Summary

## Original Test Case
The original test case from `origin/source.sv` contains:
- Array declaration
- Function call
- Always_comb block with function call
- Immediate assertion with $error() and format string: `$error("Assertion failed: arr[%0d] != 1", idx)`

This generates MLIR with:
- `sim.fmt.literal "Error: Assertion failed: arr["`
- `sim.fmt.literal "] != 1"`
- `sim.fmt.dec %idx specifierWidth 0`
- `sim.fmt.concat (%0, %9, %1)`
- `sim.proc.print %10`

All inside an `llhd.combinational` region.

## Minimization Attempts

### Attempt 1: Remove array and function logic
**Result**: Still crashes with same error

The assertion with format string is the trigger. Removing the array doesn't affect the issue.

### Attempt 2: Simplify to just assertion
**Result**: Still crashes with same error

Even the simplest assertion still creates `sim.fmt.literal` operations and fails.

### Attempt 3: Use $display instead of $error
**Result**: Different error - `llhd.process` legalization failure

Display statements also fail, but with a different error message.

## Minimal Reproduction Case

The minimal case that reproduces the **original bug** is:

```systemverilog
module test_minimal(
  input logic clk
);

initial begin
  assert (1'b0 == 1'b1) else $error("Assertion failed");
end

endmodule
```

**Reproduction Command**:
```bash
export PATH=/opt/llvm-22/bin:$PATH && \
circt-verilog --ir-hw test_minimal.sv | \
arcilator | \
opt -O0
```

**Expected Behavior**: The assertion with format string should compile successfully.

**Actual Behavior**: Fails with `failed to legalize operation 'sim.fmt.literal'`

## Root Cause Confirmed

The bug is in the ArcToLLVM conversion pass handling of `sim.fmt.literal` operations that:
1. Are created from SystemVerilog assertions
2. Appear inside `arc.execute` regions after conversion from `llhd.combinational`
3. Are marked as LEGAL in the conversion target but fail to legalize

This appears to be an incomplete or buggy implementation of feature added in commit f40496973 (Jan 19, 2026) for supporting sim.proc.print and sim.fmt.* operations.
