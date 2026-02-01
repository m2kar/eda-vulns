# Test Case Minimization Log

## Original Test Case
- File: `test.sv`
- Lines: 18
- Features: inout port, tri-state assignment, parameterized array, loop

## Minimization Strategy
Based on root cause analysis, the key triggers are:
1. **inout port** with tri-state assignment (`1'bz`)
2. Parameterized array (`logic [P1-1:0]`)

Minimization will proceed by iteratively removing features:
1. Remove parameterized array and loop
2. Remove output port
3. Keep minimal: input + inout with tri-state
4. Verify if minimal still triggers issue

## Iteration 1: Remove parameter and loop

## Test Results

All minimal test cases compiled successfully on current toolchain:

### minimal_1.sv (6 lines)
```systemverilog
module MixPorts(
  input  logic a,
  inout  wire  c
);
  assign c = a ? 1'bz : 1'b0;
endmodule
```
**Result:** ✅ Compiled successfully (no crash)

### minimal_2.sv (5 lines)
```systemverilog
module MixPorts(
  inout  wire  c
);
  assign c = 1'bz;
endmodule
```
**Result:** ✅ Compiled successfully (no crash)

### minimal_3.sv (6 lines)
```systemverilog
module MixPorts(
  input  logic a,
  inout  wire  c
);
  assign c = 1'b0;
endmodule
```
**Result:** ✅ Compiled successfully (no crash)

## Minimization Conclusion

Since the original test case does not crash on the current toolchain (CIRCT firtool-1.139.0), minimization cannot proceed based on crash reproduction. However, based on the root cause analysis, the minimal reproducer that would have triggered the crash in the original environment is:

### Minimal Reproducer (minimal_1.sv)

**Key Elements:**
- Input port (logic)
- Inout port (wire)
- Tri-state assignment to inout port (1'bz)

This test case captures all the essential features that likely triggered the LLHD reference type issue in the original arcilator.

**Note:** The bug appears to have been fixed in the current CIRCT build, as neither the original test case nor any minimized variant crashes.
