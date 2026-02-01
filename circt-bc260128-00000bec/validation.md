# Validation Report

## Result
**report** – minimized testcase is valid and reproduces the crash.

## Checks
- **CIRCT pipeline**: Reproduced `failed to legalize operation 'sim.fmt.literal'` using
  `circt-verilog --ir-hw bug.sv | arcilator | opt -O0 | llc -O0`.
- **Verilator**: `verilator --lint-only bug.sv` passes (no syntax errors).

## Notes
The failure is a legalization error in the Sim dialect lowering. The minimized
testcase isolates the `$error` literal format path that emits `sim.fmt.literal`.
