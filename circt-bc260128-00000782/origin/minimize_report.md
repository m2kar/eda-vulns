# Minimize Report

## Summary
- **Original File**: source.sv (471 bytes, 22 lines)
- **Minimized File**: bug.sv (72 bytes, 3 lines)
- **Reduction**: 84.7%

## Minimization Process

### Iteration 1: Remove always_ff block
- Removed clocked counter logic
- Changed `idx` from internal register to input port
- **Result**: Crash reproduced ✓

### Iteration 2: Simplify to single signal
- Removed array `arr`, replaced with single bit `y`
- **Result**: Crash reproduced ✓

### Iteration 3: Remove output and assignment
- Kept only input `x` and assertion
- **Result**: Crash reproduced ✓

### Iteration 4: Single-line format (FINAL)
- Compressed to minimal form: `always_comb assert (x) else $error("f");`
- **Result**: Crash reproduced ✓

### Additional Discovery
- Assert without `else` clause triggers different error (`verif.assert`)
- `$display()` in `always_comb` triggers same `sim.fmt.literal` error
- Root cause: arcilator lacks support for `sim.fmt.literal` in combinational contexts

## Final Minimized Testcase

```systemverilog
module m(input x);
  always_comb assert (x) else $error("f");
endmodule
```

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv | arcilator
```

## Error Signature

```
<stdin>:3:10: error: failed to legalize operation 'sim.fmt.literal'
    %0 = sim.fmt.literal "Error: f"
         ^
```

## Key Constructs Preserved
1. `always_comb` block (combinational context)
2. Immediate assertion (`assert`)
3. `$error()` system task with format string

## Conclusion
The crash is triggered by any format string operation (`$error`, `$display`, etc.) within an `always_comb` block when processed through arcilator. The `sim.fmt.literal` operation is not properly lowered to LLVM in the LowerArcToLLVM pass.
