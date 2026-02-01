# Validation Report

## Syntax Check
* Tool: `circt-verilog --ir-hw bug.sv`
* Exit code: 0
* Result: ✅ Parsed successfully

## Cross-Tool Validation
* Verilator: not checked
* Slang: not checked

## Classification
Result: `report`

Rationale: The testcase is syntactically valid and the failure occurs during the
arcilator pipeline (arc.state type mismatch / assertion in InferStateProperties),
indicating a CIRCT bug rather than an invalid testcase.
