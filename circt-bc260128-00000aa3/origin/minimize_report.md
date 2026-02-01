# Minimization Report

## Summary

| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| File size (bytes) | 606 | 191 | 68.5% |
| Lines of code | 30 | 9 | 70% |

## Reduction Steps

### Step 1: Reduce array size (16 → 2)
- Changed `logic [7:0] arr [0:15]` to `logic [7:0] arr [0:1]`
- Result: **Still crashes** ✓

### Step 2: Remove intermediate sum variable
- Replaced `logic [11:0] sum` with direct computation
- Result: **Still crashes** ✓

### Step 3: Remove loop in always_comb
- Simplified combinational sum to direct expression
- Result: **Still crashes** ✓

### Step 4: Remove data_in input
- Changed `arr[0] <= data_in` to `arr[0] <= constant`
- Result: **Still crashes** ✓

### Step 5: Combine always_ff blocks
- Merged two separate always_ff blocks into one
- Result: **Still crashes** ✓

### Step 6: Simplify element width (8 → 1 bit)
- Changed `logic [7:0] arr` to `logic arr`
- Result: **Still crashes** ✓

### Step 7: Simplify module name and variable names
- Changed `array_reg` to `m`, `arr` to `a`
- Result: **Still crashes** ✓

## Essential Elements for Crash

The minimized test case demonstrates these **required elements**:

1. **Unpacked array**: `logic a [0:1]` - Array type triggers the bug
2. **Loop in registered block**: `for (int i = 1; i < 2; i++) a[i] <= a[i-1]` - Creates the shift pattern
3. **Registered output referencing array**: `out <= a[0]` - Creates enable pattern detection

## What Could Be Removed

- Original module name (`array_reg`)
- Input `data_in` (constant works)
- Combinational `sum` variable
- Loop in `always_comb`
- Second `always_ff` block (can be combined)
- Larger array size (2 elements sufficient)
- Larger element width (1 bit sufficient)

## Minimized Test Case

```systemverilog
module m(input clk, output logic out);
  logic a [0:1];
  always_ff @(posedge clk) begin
    out <= a[0];
    a[0] <= 0;
    for (int i = 1; i < 2; i++)
      a[i] <= a[i-1];
  end
endmodule
```

## Crash Signature

The crash manifests as an MLIR type mismatch error:
```
error: 'arc.state' op operand type mismatch: operand #2
expected type: '!hw.array<2xi1>'
  actual type: 'i<garbage>'  (e.g., i673700224)
```

The garbage integer type (like `i673700224`) indicates uninitialized memory being read as a type value, consistent with the root cause analysis showing that `hw::ConstantOp::create` is called with an array type instead of an integer type.
