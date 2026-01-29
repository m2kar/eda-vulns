# Minimize Report

## Summary
- **Original size**: 17 lines (339 bytes)
- **Minimized size**: 10 lines (316 bytes)
- **Reduction**: 41% (lines), 7% (bytes)
- **Crash verified**: ✅ Yes

## Key Constructs Preserved
Based on `analysis.json`, the following critical constructs were preserved:
1. ✅ `packed struct` - Essential for triggering the bug
2. ✅ `packed array` - Required (struct array `t [1:0] a`)
3. ✅ `bit-level indexing` - Critical (`a[0].f[0]`)
4. ✅ `always_comb` - Required context for the bug

## Minimization Steps

### Step 1: Baseline verification
Original test case crashes as expected.

### Step 2: Simplify declarations
- Reduced module name: `top_module` → `m`
- Reduced port names: `clk`, `D`, `Q` simplified
- Reduced struct type name: `array_elem_t` → `t`
- Reduced field name: `field0` → `f`

### Step 3: Reduce array and field sizes
- Array size: `[7:0]` → `[1:0]` (8 elements → 2 elements)
- Field width: `logic [4:0]` → `logic [1:0]` (5 bits → 2 bits)
- ✅ Crash still reproduces

### Step 4: Remove always_ff
- Replaced `always_ff @(posedge clk) Q <= ...` with usage in `always_comb`
- Removed clock port
- ✅ Crash still reproduces

### Step 5: Try single bit field (FAILED)
- Changed to `logic f` (1 bit)
- ❌ Crash does NOT reproduce - bit-level indexing is required

### Step 6: Final minimization
- Combined write and read in same `always_comb` block
- ✅ Minimal test case confirmed

## Minimal Test Case
```systemverilog
module m(input logic D, output logic Q);
  typedef struct packed { logic [1:0] f; } t;
  t [1:0] a;
  always_comb begin
    a[0].f[0] = D;
    Q = a[0].f[0];
  end
endmodule
```

## Crash Signature
```
Assertion `op->use_empty() && "expected 'op' to have no uses"' failed
Location: PatternMatch.cpp:156 (mlir::RewriterBase::eraseOp)
Triggered by: extractConcatToConcatExtract in CombFolds.cpp:548
```

## Critical Findings
1. **Minimum array size**: 2 elements (`[1:0]`)
2. **Minimum field width**: 2 bits (`logic [1:0]`)
3. **Required constructs**:
   - Packed struct array with bit-level indexing
   - `always_comb` block (not `assign`)
   - Same indexed location must be both written and read
