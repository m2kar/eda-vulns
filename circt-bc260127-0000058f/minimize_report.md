# Test Case Minimization Report

## Summary

| Metric | Value |
|--------|-------|
| Original Lines | 24 |
| Final Lines | 9 |
| Reduction | 62.5% |
| Original File | source.sv |
| Minimized File | bug.sv |

## Minimization Process

### Step 1: Analyze Key Construction
From root cause analysis:
- **Trigger**: Packed union type as module port
- **Key construction**: `typedef union packed { ... } u; module m(input u x);`

### Step 2: Reduction Iterations

1. **Initial test (8 lines)** - Matched key_construction pattern exactly
   - Result: CRASH ✓
   
2. **Remove second union field** (6 lines)
   - Reduced from 2 fields to 1 field
   - Result: CRASH ✓
   
3. **Simplify field type** (6 lines)  
   - Changed `logic [7:0]` to `logic`
   - Result: CRASH ✓

4. **Try inline union in port** (2 lines)
   - `module m(input union packed {logic a;} x);`
   - Result: CRASH ✓ (but less readable)

### Step 3: Final Minimized Case
Chose 9-line version with comments for clarity:

```systemverilog
// Minimal reproducer: packed union type as module port crashes MooreToCore pass
// Bug: Missing type conversion for PackedUnionType in MooreToCore

typedef union packed {
  logic a;
} u;

module m(input u x);
endmodule
```

## Elements Removed

| Element | Original | Minimized | Reason |
|---------|----------|-----------|--------|
| Module body | 9 lines (clk, rst, always_ff, assign) | 0 | Body not needed - crash on port type |
| Union fields | 2 fields | 1 field | Only 1 field needed to trigger |
| Output port | 1 output | 0 | Input only sufficient |
| Control ports | clk, rst_n | none | Not needed for type conversion bug |
| Registers | data_reg | none | Not used |

## Reproduction Command
```bash
circt-verilog --ir-hw bug.sv
```

## Verification
Minimized case reproduces same crash:
- **Crash type**: Assertion failure
- **Assertion**: `detail::isPresent(Val) && "dyn_cast on a non-existent value"`
- **Stack trace**: MooreToCore.cpp → SVModuleOpConversion::matchAndRewrite
