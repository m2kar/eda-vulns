# Validation Report

## Summary

| Property | Value |
|----------|-------|
| **Validation Result** | `report` |
| **Syntax Valid** | ✅ Yes |
| **Cross-Tool Check** | ✅ Passed |
| **Reduction** | 43.4% (267 → 151 bytes) |

## Test Case Details

### Minimized Test Case (`bug.sv`)

```systemverilog
module test(
  input logic clk,
  input real in_real,
  output real out_real
);
  always @(posedge clk) begin
    out_real <= in_real;
  end
endmodule
```

### Reproduction Command

```bash
circt-verilog --ir-hw bug.sv
```

## Cross-Tool Validation

### Verilator
- **Result**: ✅ Passed
- **Output**: No errors (lint-only mode)

### Slang
- **Result**: ✅ Passed
- **Output**: `Build succeeded: 0 errors, 0 warnings`

## Bug Classification

This is a **valid bug report** because:

1. **Valid SystemVerilog**: The test case uses valid SystemVerilog constructs:
   - `real` type is a standard IEEE 1800 floating-point type
   - Clocked assignment using `always @(posedge clk)` is standard

2. **Cross-Tool Validation**: Both Verilator and Slang accept this code without errors, confirming it's valid SystemVerilog

3. **Compiler Crash**: circt-verilog crashes with an assertion failure instead of producing a graceful error message

4. **Root Cause**: The LLHD Mem2Reg pass calls `hw::getBitWidth()` which returns -1 for unsupported types like `real`. This -1 is interpreted as an unsigned value (~4 billion), causing an attempt to create an `IntegerType` with a bitwidth exceeding MLIR's 16,777,215 bit limit.

## Conclusion

**Result: `report`**

This is a valid bug that should be reported to the CIRCT project. The compiler should either:
1. Support `real` types properly in the LLHD dialect
2. Reject `real` types with a graceful error message before reaching the Mem2Reg pass
3. Handle the -1 return value from `getBitWidth()` gracefully instead of crashing
