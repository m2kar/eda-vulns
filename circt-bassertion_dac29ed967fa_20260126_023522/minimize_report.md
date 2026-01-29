# Minimize Report

## Summary
- **Original Size**: 274 bytes (17 lines)
- **Minimized Size**: 45 bytes (2 lines)
- **Reduction**: 83.6%

## Minimized Test Case
```systemverilog
module test(output string result);
endmodule
```

## Minimization Process

### Step 1: Remove unrelated constructs
- Removed `reg r1` and `always_ff` block (unrelated to string port)
- Removed `initial` block (string initialization)
- Removed `$display` statement
- **Result**: Crash still reproducible

### Step 2: Remove internal variables
- Removed `string s` internal variable
- Removed `always_comb` block
- **Result**: Crash still reproducible

### Step 3: Simplify module ports
- Removed `input logic clk` (not needed)
- Kept only `output string result`
- **Result**: Crash still reproducible

### Step 4: Verify minimal form
- Tested `input string s` - also crashes
- Confirmed: Any module with string type port triggers the crash

## Key Finding
The crash is triggered by the mere presence of a `string` type port in a module declaration. No module body is required.

## Reproduction Command
```bash
circt-verilog --ir-hw bug.sv
```

## Crash Signature
- **Type**: Assertion failure
- **Location**: MooreToCore.cpp, SVModuleOpConversion::matchAndRewrite
- **Root Cause**: String type ports are not properly handled during Moore to Core conversion
