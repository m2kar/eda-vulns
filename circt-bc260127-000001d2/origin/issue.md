# [arcilator] Legalization failure: orphaned sim.fmt.literal from immediate assertions with $error()

## Summary

Immediate assertions with `$error()` in `always_comb` blocks cause a legalization failure in `arcilator`'s `LowerArcToLLVM` pass. The `sim.fmt.literal` operations generated from error messages remain unconverted and trigger "failed to legalize operation 'sim.fmt.literal'".

## Testcase

```systemverilog
module m(input q);
  always_comb assert(q) else $error("");
endmodule
```

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv | arcilator
```

## Error

```
<stdin>:3:10: error: failed to legalize operation 'sim.fmt.literal'
    %0 = sim.fmt.literal "Error: "
         ^
<stdin>:3:10: note: see current operation: %7 = "sim.fmt.literal"() <{literal = "Error: "}> : () -> !sim.fstring
```

## Root Cause Analysis

The `sim.fmt.literal` operation created from `$error()` in immediate assertions is orphaned - it's not properly consumed by a `PrintFormattedProcOp` or removed by DCE before the legalization check.

### Technical Details

1. **IR Generation**: `circt-verilog --ir-hw` converts the `$error()` call to a `sim.fmt.literal` operation

2. **LowerArcToLLVM Pass Design** (from `lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp`):
   - Lines 1234-1239: `sim::FormatLiteralOp` is marked as **legal** (not to be converted)
   - The design expects format literals to be consumed by `SimPrintFormattedProcOpLowering` via `foldFormatString()`
   - After conversion, the `Pure` trait should allow DCE to remove unused format ops

3. **The Bug**: When processing immediate assertions:
   - Format literals are generated without being connected to a print operation
   - OR the print operation is lowered separately, leaving format literals orphaned
   - OR there's a missing conversion pattern for assertions with message output in the arcilator pipeline

4. **Why Legalization Fails**:
   - Despite being marked "legal", the operation still triggers a legalization error
   - This suggests the operation is present in an unexpected context or the legalization target isn't properly configured for this pipeline path

## Related Issues

- **#9467**: Similar legalization failure pattern with orphaned `llhd.constant_time` operations
  - Same arcilator/LowerArcToLLVM context
  - Same root cause pattern (orphaned operation not properly consumed)
  - Different dialect (LLHD vs Sim) and different operation type

## Cross-Tool Validation

- **Verilator**: `lint-only passed with no errors`
- **Slang**: `Build succeeded: 0 errors, 0 warnings`
- **circt-verilog**: Parses correctly and generates valid MLIR with `sim.fmt.literal` linked to `sim.proc.print`

This confirms the SystemVerilog syntax is valid and the crash is a bug in arcilator's legalization handling.

## Affected Components

- `lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp` - FormatLiteralOp marked legal at lines 1236-1239
- `lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp` - foldFormatString handling at lines 805-823
- `lib/Conversion/ImportVerilog/Statements.cpp` - Assertion import handling
- `lib/Dialect/Sim/SimOps.cpp` - FormatLiteralOp implementation

## Possible Fixes

1. **Add assertion lowering in arcilator**: Create a lowering pattern for immediate assertions that properly handles `$error()` message output

2. **Ensure format literals are always consumed**: Modify assertion lowering to wrap format literals in a print operation

3. **Add cleanup pass**: Run a pass before LowerArcToLLVM that removes orphaned format strings or converts them to no-ops

4. **Add legalization pattern**: Provide a lowering pattern that emits a warning/no-op for orphaned format literals instead of failing

## Environment

- CIRCT version: built from source
- LLVM version: 22.0.0git
- Reproduction: deterministic (100%)
- Testcase ID: 260127-000001d2
