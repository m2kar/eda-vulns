# Minimize Report

## Summary
Successfully minimized test case from 14 lines to 2 lines (85.7% reduction).

## Original Test Case
```systemverilog
module Mod(input logic [1:0] a, output string str_out);
  string str = "test";

  always_comb begin
    if (a == 2'b00)
      str_out = str;
    else
      str_out = "default";
  end
endmodule

module Top;
  string my_str;
  Mod inst1(.a(2'b01), .str_out(my_str));
endmodule
```
**Lines:** 14 (excluding empty lines)

## Minimized Test Case
```systemverilog
module Mod(output string str_out);
endmodule
```
**Lines:** 2

## Reduction Statistics
| Metric | Value |
|--------|-------|
| Original Lines | 14 |
| Minimized Lines | 2 |
| Reduction | 85.7% |

## Minimization Process

### Step 1: Identify Key Constructs
From `analysis.json`:
- `string type as module output port` ← **CRITICAL**
- `string variable declaration`
- `string literal assignment`
- `module instantiation with string port`

### Step 2: Remove Non-Essential Code
1. Removed `input logic [1:0] a` - not related to crash
2. Removed `string str = "test";` - internal variable not needed
3. Removed `always_comb` block - assignment logic not needed
4. Removed `module Top` - instantiation not needed

### Step 3: Verify Crash Signature
**Original signature:**
```
SVModuleOpConversion::matchAndRewrite MooreToCore.cpp dyn_cast on a non-existent value
```

**Minimized signature:**
```
SVModuleOpConversion::matchAndRewrite MooreToCore.cpp
```

✅ Crash signature matches (same function and file)

### Step 4: Additional Testing
- Tested `input string` - also crashes ✅
- Tested empty module body - still crashes ✅
- Cannot reduce further - module declaration and string port are minimal

## Root Cause Correlation
The minimized test case confirms the hypothesis in `analysis.json`:
- The crash is triggered specifically by `string` type in module port
- No module body or other constructs are needed
- The issue is in type conversion during `MooreToCore` pass when handling `StringType` ports

## Command to Reproduce
```bash
circt-verilog --ir-hw bug.sv
```
