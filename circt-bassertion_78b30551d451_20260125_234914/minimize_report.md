# Minimization Report

## Summary
- **Original file**: source.sv (10 lines)
- **Minimized file**: bug.sv (2 lines)
- **Reduction**: 80%
- **Crash preserved**: Yes

## Key Construct Preserved

Based on `analysis.json`, the trigger construct is:
- `input string a` - dynamic string type as module port

## Original vs Minimized

### Original (source.sv)
```systemverilog
module top(input string a, output logic [7:0] out);
  int length;
  logic [7:0] in = 8'hFF;

  always_comb begin
    length = a.len();
  end
  
  assign out[7-:4] = in;
endmodule
```

### Minimized (bug.sv)
```systemverilog
module top(input string a);
endmodule
```

## Removed Elements
- Output port `out` (not needed for crash)
- Variable declarations `length`, `in`
- `always_comb` block with `a.len()` call
- `assign` statement

## Verification

### Crash Signature Match
- **Original**: `SVModuleOpConversion::matchAndRewrite` crash in MooreToCore.cpp
- **Minimized**: Same crash location confirmed

### Exit Code
- Exit code: 139 (SIGSEGV)

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv
```
