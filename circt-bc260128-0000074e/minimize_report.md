# Minimization Report

## Summary
- Original file: origin/source.sv (22 lines)
- Minimized file: bug.sv (10 lines)
- Reduction: 60.0%

## Minimization Steps

1. Removed submodule `submod` and its instantiation - **still hangs**
2. Removed extra output `shared_data` - **still hangs**
3. Simplified enum to 2 states - **still hangs**
4. Replaced enum with plain `logic` - **still hangs**
5. Simplified struct to 2 single-bit fields - **still hangs**
6. Used `if` instead of `case` - **still hangs**
7. Tried array output instead of struct - **WORKS** (no hang)
8. Used ternary operator - **still hangs**
9. Used direct assignment `out.b = out.a` - **still hangs** ✓ MINIMAL

## Key Finding
The bug requires:
1. **Packed struct** as output port (array doesn't trigger it)
2. **always_comb** block (continuous assign doesn't trigger it)
3. Reading one field while writing another of the **same output struct**

## Minimal Reproducer
```systemverilog
typedef struct packed { logic a; logic b; } s_t;

module top(output s_t out);
  always_comb begin
    out.b = out.a;  // Read from out.a, write to out.b -> causes infinite loop
  end
endmodule
```
