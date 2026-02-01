# Validation Report

## CIRCT
- `circt-verilog --ir-hw bug.sv` crashes with assertion (see error.log)

## Verilator
Command:
```
verilator --lint-only -Wall bug.sv
```
Result: `DECLFILENAME` warning (module name `test` vs file name `bug`), no crash.

## Slang
Command:
```
slang --lint-only bug.sv
```
Result: Build succeeded (0 errors, 0 warnings).

## Classification
This is a CIRCT crash; testcase is otherwise valid SV.
