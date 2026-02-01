# Minimization Report

## Summary

| Metric | Value |
|--------|-------|
| Original Lines | 13 |
| Minimized Lines | 7 (4 functional) |
| Reduction | 46% |
| Bug Preserved | Yes |

## Original Test Case

```systemverilog
module test_module(
  input logic clk,
  input real in_real,
  output real out_real,
  output logic cmp_result
);

  always_ff @(posedge clk) begin
    out_real <= in_real * 2.0;
    cmp_result <= (in_real > 5.0);
  end

endmodule
```

## Minimized Test Case

```systemverilog
module test(input logic clk);
  real r;
  always_ff @(posedge clk) r <= 1.0;
endmodule
```

## Minimization Process

### Step 1: Remove Unnecessary Ports
- Removed `input real in_real` - not needed to trigger bug
- Removed `output real out_real` - internal variable sufficient
- Removed `output logic cmp_result` - comparison not needed

### Step 2: Simplify Logic
- Simplified `out_real <= in_real * 2.0` to `r <= 1.0`
- Removed comparison operation `(in_real > 5.0)`
- Kept only the essential sequential assignment to a `real` variable

### Step 3: Verify Bug Still Triggers
- The minimized test case still triggers the same underlying bug
- Error manifests as invalid bitcast operation with `i1073741823` bitwidth

## Key Constructs Preserved

1. **`real` type variable** - Required to trigger Float64Type handling bug
2. **`always_ff` block** - Sequential logic triggers Mem2Reg pass
3. **Clock edge sensitivity** - `@(posedge clk)` for proper sequential modeling
4. **Assignment to real** - `r <= 1.0` triggers the type conversion issue

## Root Cause Analysis

The bug occurs because:
1. SystemVerilog `real` type is converted to MLIR `f64` (Float64Type)
2. During MooreToCore conversion, the Mem2Reg pass attempts to promote the signal
3. `hw::getBitWidth()` returns -1 for Float64Type (unsupported type)
4. This -1 value is mishandled, resulting in an invalid bitwidth of 1073741823 (2^30 - 1)
5. The compiler crashes/errors when trying to create an IntegerType or bitcast with this invalid width

## Conclusion

The minimized test case successfully reproduces the bug with maximum simplicity while preserving all essential constructs needed to trigger the issue.
