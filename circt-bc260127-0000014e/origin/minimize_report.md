# Minimization Report

## Summary
- **Original file:** `source.sv` (7 lines)
- **Minimized file:** `bug.sv` (2 lines)
- **Reduction:** 5 lines removed (71.43%)
- **Crash status:** Still reproduces the same `dyn_cast on a non-existent value` assertion.

## Rationale
The root cause points to an unsupported **string input port** during Moore-to-Core
type conversion. The crash happens while sanitizing port types, so only the module
declaration and the `input string` port are necessary to trigger the failure.

## Reduction steps
1. Removed wire declarations and assignments unrelated to port type conversion.
2. Kept the module declaration structure with the `input string` port and output.
3. Verified the minimal module still crashes in `circt-verilog --ir-hw`.

## Minimized test case
```systemverilog
module test(input string a, output int b);
endmodule
```
