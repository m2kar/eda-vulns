# Minimization Report

## Summary

Successfully minimized test case from 10 lines to 2 lines (80% reduction).

## Original Test Case (source.sv)

```systemverilog
module test(input string a, output int b);
  logic [7:0] arr [0:3];
  int idx;
  
  always_comb begin
    idx = a.len();
    arr[idx] = 8'hFF;
    b = idx;
  end
endmodule
```

## Minimized Test Case (bug.sv)

```systemverilog
module test(input string a);
endmodule
```

## Minimization Steps

1. **Identified Root Cause**: The crash is triggered by `input string a` - a string type port in the module interface.

2. **Removed Components**:
   - ✅ Removed `output int b` - Not needed for crash
   - ✅ Removed `logic [7:0] arr [0:3]` - Array declaration not needed
   - ✅ Removed `int idx` - Variable not needed
   - ✅ Removed entire `always_comb` block including:
     - `idx = a.len()` - String method call not needed
     - `arr[idx] = 8'hFF` - Array assignment not needed
     - `b = idx` - Output assignment not needed

3. **Verification**: The minimal test case `module test(input string a); endmodule` still crashes with the same stack trace at `SVModuleOpConversion::matchAndRewrite` in `MooreToCore.cpp`.

## Root Cause Confirmation

The crash occurs during MooreToCore conversion when the `string` input port type (`moore::StringType`) is converted to `sim::DynamicStringType`. This converted type is incompatible with the HW dialect's port infrastructure, causing an assertion failure in `hw::ModulePortInfo` construction.

The minimal test proves that:
- No usage of the string variable is required
- No output ports are required  
- No module body is required
- Simply declaring a module with a string input port is sufficient to trigger the bug
