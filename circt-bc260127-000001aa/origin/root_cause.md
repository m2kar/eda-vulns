# Root Cause Analysis

## Summary
The crash is an MLIR assertion triggered during the LLHD Sig2Reg pass when it tries to RAUW a value with itself.

## Evidence
- Assertion: `cannot RAUW a value with itself` from `mlir/IR/UseDefLists.h:213`.
- Stack trace points to `lib/Dialect/LLHD/Transforms/Sig2RegPass.cpp`:
  - `SigPromoter::isPromotable()` and `Sig2RegPass::runOnOperation()`.
  - The RAUW happens in `SigPromoter::promote()` when replacing reads.

## Likely Cause
The test case creates a simple combinational loop (`internal_wire` drives `out`, and `out` drives `internal_wire`).
During LLHD signal promotion, `SigPromoter::promote()` materializes a `read` value and then calls
`interval.value.replaceAllUsesWith(read)`. For this pattern, the `read` SSA value aliases the
same underlying use list as `interval.value`, causing an illegal self-RAUW.

## Affected Code
- `lib/Dialect/LLHD/Transforms/Sig2RegPass.cpp`:
  - `SigPromoter::promote()` replaces read uses.
  - Self-RAUW assertion in `mlir/IR/UseDefLists.h:213`.

## Notes
This reproduces with `/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw source.sv`.
The `/opt/firtool/bin/circt-verilog` build hangs on the same input.
