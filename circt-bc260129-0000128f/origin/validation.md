# Validation Report

## Result
**report**

## Command
```
circt-verilog --ir-hw bug.sv | arcilator | opt -O0 | llc -O0 --filetype=obj -o /tmp/arcilator_repro.o
```

## Outcome
- Exit code: 1
- Diagnostic: `'arc.state' op operand type mismatch`

## Notes
The current toolchain fails with a verifier error instead of asserting, but the
failure remains in the Arc pipeline and is likely the same root issue.
