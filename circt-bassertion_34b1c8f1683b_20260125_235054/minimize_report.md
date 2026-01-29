# Minimization Report

## Summary
- **Original file**: source.sv (22 lines)
- **Minimized file**: bug.sv (2 lines)
- **Reduction**: 90.9%
- **Crash preserved**: Yes

## Preservation Analysis

### Key Constructs Preserved
Based on `analysis.json`, the critical construct was identified as:
- `string` type as module port

### Original Code Structure
```systemverilog
module test_module(
  input logic clk,
  input logic rst,
  output string str_out  // <-- KEY: string type port
);
  string str;
  initial begin ... end
  always @(posedge clk) begin ... end
  assign str_out = str;
endmodule
```

### Minimized Code
```systemverilog
module m(output string s);
endmodule
```

### Removed Elements
- `input logic clk` - not needed for crash
- `input logic rst` - not needed for crash
- `string str` internal variable - not needed for crash
- `initial` block - not needed for crash
- `always` block - not needed for crash
- `assign` statement - not needed for crash
- Module renamed from `test_module` to `m` for brevity
- Port renamed from `str_out` to `s` for brevity

## Verification

### Original Assertion (from error.txt)
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Minimized Crash Location
```
SVModuleOpConversion::matchAndRewrite MooreToCore.cpp
```

**Match**: The crash occurs in the same location (`MooreToCore.cpp` during module conversion).

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv
```

## Notes
- The crash is triggered by the mere presence of a `string` type port
- No other code constructs are necessary to reproduce
- Both `input string` and `output string` trigger the same crash
- The issue is in the MooreToCore conversion pass which lacks type conversion rules for `StringType`
