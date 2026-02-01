# Minimize Report

## Summary

Successfully minimized the testcase from 22 lines to 5 lines (77% reduction).

## Original Testcase

```systemverilog
module top_module(input logic clk, input logic resetn);
  
  // Class definition with typedef referencing itself in parameterized template
  class registry #(type T = int);
    // Parameterized registry class
  endclass
  
  class my_class;
    typedef registry#(my_class) type_id;
  endclass
  
  // Instance of the class
  my_class obj;
  
  // Sequential logic with clock edge
  always_ff @(posedge clk) begin
    if (resetn) begin
      obj = new();
    end
  end
  
endmodule
```

## Minimized Testcase

```systemverilog
module top(input logic clk);
  class C; endclass
  C obj;
  always_ff @(posedge clk) obj = new();
endmodule
```

## What Was Removed

1. **Unnecessary module ports**: Removed `resetn` port - the condition was not essential for triggering the bug
2. **Parameterized class template**: Removed `registry #(type T = int)` class - not needed
3. **Self-referencing typedef**: Removed `typedef registry#(my_class) type_id` - not needed
4. **Verbose formatting**: Consolidated to single-line statements where appropriate
5. **Comments and extra whitespace**: Removed all comments

## What Was Preserved (Essential Constructs)

1. **Class definition**: Minimal `class C; endclass`
2. **Class handle variable**: `C obj;` - triggers ClassHandleType creation
3. **always_ff block**: `always_ff @(posedge clk)` - triggers Mem2Reg pass
4. **new() call**: `obj = new();` - creates instance requiring type handling

## Key Finding

The original analysis identified self-referencing typedef as the key construct, but further minimization revealed that **any class handle used in always_ff** triggers the same underlying bug. The parameterized template and typedef were incidental to the original testcase, not essential to the bug.

## Root Cause Relationship

The minimized testcase still exercises the same code path:
- `ClassHandleType` is created for variable `obj`
- `always_ff` causes Mem2Reg pass to run
- `hw::getBitWidth()` returns -1 for `ClassHandleType`
- The value 1073741823 (0x3FFFFFFF) in the error output is derived from this invalid bitwidth
- Error occurs when trying to create `hw.bitcast` with incompatible types

## Reproduction Command

```bash
/opt/firtool/bin/circt-verilog --ir-hw bug.sv
```

## Error Signature

```
error: 'hw.bitcast' op result #0 must be Type wherein the bitwidth in hardware is known, but got '!llvm.ptr'
note: see current operation: %10 = "hw.bitcast"(%9) : (i1073741823) -> !llvm.ptr
```
