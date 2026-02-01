# Minimize Report

## Summary

| Metric | Value |
|--------|-------|
| Original Lines | 27 (with EOF newline: 26 actual code) |
| Minimized Lines | 14 |
| Reduction | 48% reduced (from 27 to 14 lines) |
| Crash Preserved | ✅ Yes |

## Key Constructs Identified

From `analysis.json`:
- `packed_struct`: ✅ Retained
- `struct_array`: ✅ Retained  
- `sequential_shift_register`: ✅ Retained (for-loop pattern)
- `posedge_clock`: ✅ Retained

## Minimization Steps

1. **v1**: Removed `xor_value` and `result_out` calculations → No crash (output needed)
2. **v2**: Simplified output to just `shift_reg[3].valid` → **Crash**
3. **v3**: Reduced array size to 2, removed for-loop → No crash
4. **v4**: Array size 3 with for-loop → **Crash**
5. **v5**: Single-field struct (8-bit data) → **Crash**
6. **v6**: Minimal 1-bit struct → **Crash**
7. **v7**: Inline struct (no typedef) → **Crash**
8. **v8**: Array size 2 with for-loop → **Crash**
9. **v9**: Without for-loop (explicit assignment) → No crash
10. **v10**: Removed initialization → **Crash** (Final candidate)

## Critical Trigger Pattern

The crash requires ALL of these:
1. **Struct array** (packed or unpacked)
2. **For-loop** with shift pattern `s[i] <= s[i-1]`
3. **Output port** reading from the array

Without the for-loop (using explicit assignments), the same logic does not crash.

## Final Minimized Testcase

```systemverilog
module test(
  input logic clk,
  output logic o
);
  struct packed { logic d; } s[2];

  always @(posedge clk) begin
    for (int i = 1; i < 2; i++)
      s[i] <= s[i-1];
    o <= s[1].d;
  end
endmodule
```

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv | arcilator
```

## Crash Signature

```
Assertion `isa<To>(Val) && "cast<Ty>() argument of incompatible type!"' failed.
```

Location: `InferStateProperties.cpp:211` in `applyEnableTransformation`
