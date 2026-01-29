# Test Case Minimization Report

## Original Test Case

```systemverilog
module test_module(input logic clk, input string a);
  logic r1;
  int b;
  
  always @(posedge clk) begin
    b = a.len();
    r1 = (b > 0) ? 1'b1 : 1'b0;
  end
endmodule
```

**Lines**: 9
**Key Constructs**: module, input string port, always block, string.len() method

## Minimization Process

### Iteration 1: Remove always block and variables
**Test Case**:
```systemverilog
module test_module(input string a);
endmodule
```

**Result**: ✅ Crash reproduced

**Deleted**:
- `input logic clk` (non-essential port)
- `logic r1` (unused variable)
- `int b` (unused variable)
- `always @(posedge clk)` block (entire block)
- `b = a.len()` (string method call)
- `r1 = (b > 0) ? 1'b1 : 1'b0` (conditional assignment)

### Analysis
The crash is triggered by the presence of `input string a` alone. The string type as a module port causes the MooreToCore type conversion to fail, resulting in a null type being passed to `sanitizeInOut()` which crashes on `dyn_cast<InOutType>`.

The additional constructs (clock, always block, string methods) are not required to trigger the bug.

## Final Minimized Test Case

```systemverilog
module test_module(input string a);
endmodule
```

## Reduction Statistics
- Original: 9 lines
- Minimized: 2 lines
- Reduction: 77.78%

## Verification
✅ Crash reproduced with same signature:
- **Crash site**: `SVModuleOpConversion::matchAndRewrite` (MooreToCore.cpp)
- **Pass**: MooreToCorePass
- **Root cause**: moore::StringType has no registered type conversion, resulting in null type passed to hw::ModulePortInfo

## Reproduction Command
```bash
/opt/firtool/bin/circt-verilog --ir-hw bug.sv
```
