# Root Cause Analysis

## Summary
The crash is caused by an unhandled legalization of `sim.fmt.literal` during the
`circt-verilog --ir-hw | arcilator | opt | llc` pipeline. The SystemVerilog
`assert ... else $error("...")` generates a `sim.fmt.literal` operation in the
Sim dialect, and the downstream conversion pipeline fails because there is no
legalization/translation for this op to the target dialect (LLVM/Arcilator
runtime).

## Evidence
- Reproducer output matches the original error:
  `error: failed to legalize operation 'sim.fmt.literal'`.
- The failing op originates from the assertion message literal:
  `sim.fmt.literal "Error: Assertion failed: q_reg != 0"`.

## Hypothesis
`sim.fmt.literal` is emitted by the SV assertion `$error` formatting logic, but
the Arcilator/LLVM lowering pipeline lacks a pattern to convert it to a
lower-level representation (e.g. runtime call or constant string handling).
This leads to a legalization failure when the conversion to the LLVM dialect is
performed.

## Suggested Fix Directions
1. Add a legalization/translation pattern for `sim.fmt.literal` in the
   Sim-to-LLVM/Arcilator lowering pass.
2. Alternatively, lower `sim.fmt.literal` earlier into a supported form or
   gate the emission of `sim.fmt.literal` when targeting Arcilator.
