# Root Cause Analysis: sim.fmt.literal Legalization Failure

## Summary

**Dialect:** Sim  
**Crash Type:** Legalization Failure  
**Severity:** High  
**Reproducibility:** Deterministic

## Error Context

```
<stdin>:3:10: error: failed to legalize operation 'sim.fmt.literal'
    %0 = sim.fmt.literal "Error: Assertion failed: q != 0"
         ^
<stdin>:3:10: note: see current operation: %27 = "sim.fmt.literal"() <{literal = "Error: Assertion failed: q != 0"}> : () -> !sim.fstring
```

## Triggering Code

```systemverilog
module test_module(input logic clk, output logic q);
  always_ff @(posedge clk) begin
    q <= ~q;
  end
  
  // This immediate assertion triggers the bug
  always_comb begin
    assert (q == 1'b0) else $error("Assertion failed: q != 0");
  end
endmodule
```

## Root Cause Hypothesis

The crash occurs due to **orphaned `sim.fmt.literal` operations** that are not connected to a consuming print operation when processed by the arcilator's `LowerArcToLLVM` pass.

### Technical Analysis

1. **SystemVerilog to IR Conversion**:
   - The `$error("...")` call in an immediate assertion is converted to a `sim.fmt.literal` operation
   - This format literal is supposed to be consumed by a `sim::PrintFormattedProcOp` or similar

2. **LowerArcToLLVM Pass Design**:
   - In `lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp` (lines 1234-1239):
   ```cpp
   // Mark sim::Format*Op as legal. These are not converted to LLVM, but the
   // lowering of sim::PrintFormattedOp walks them to build up its format string.
   // They are all marked Pure so are removed after the conversion.
   target.addLegalOp<sim::FormatLiteralOp, sim::FormatDecOp, sim::FormatHexOp,
                     sim::FormatBinOp, sim::FormatOctOp, sim::FormatCharOp,
                     sim::FormatStringConcatOp>();
   ```
   
3. **The Design Assumption**:
   - Format ops are marked as **legal** (not lowered)
   - They are expected to be consumed by `SimPrintFormattedProcOpLowering` via `foldFormatString()`
   - After conversion, the `Pure` trait allows DCE (Dead Code Elimination) to remove unused format ops

4. **The Bug**:
   - When an immediate assertion with `$error()` is processed, the format literal may be generated without a corresponding `PrintFormattedProcOp` consumer
   - OR the print operation is lowered separately, leaving the format literal orphaned
   - OR there's a missing conversion pattern for assertions with message output in the arcilator pipeline
   
5. **Why Legalization Fails**:
   - Despite being marked "legal", the operation still triggers a legalization error
   - This likely indicates the operation is present in an **unexpected context** or the legalization target isn't properly configured for this pipeline path

## Evidence

| Source | Evidence | Significance |
|--------|----------|--------------|
| error.txt | `failed to legalize operation 'sim.fmt.literal'` | Confirms legalization framework rejects the op |
| error.txt | The literal contains the exact `$error()` message | Direct correlation to assertion message |
| source.sv | Immediate assertion in `always_comb` | Procedural context triggers the issue |
| LowerArcToLLVM.cpp | `addLegalOp<sim::FormatLiteralOp>` | Op should be legal but isn't in this context |

## Key Patterns

1. **Immediate assertions** (`assert ... else $error(...)`) in procedural blocks
2. **Format literals** created from `$error()` messages
3. **arcilator pipeline** processing without proper assertion handling
4. **Missing lowering** for assertion-related format strings

## Affected Components

- `lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp` - Legalization target configuration
- `lib/Conversion/ImportVerilog/Statements.cpp` - Assertion import handling
- `lib/Dialect/Sim/SimOps.cpp` - Format operation definitions

## Possible Fixes

1. **Add assertion lowering in arcilator**: Create a lowering pattern for immediate assertions that properly handles the `$error()` message output

2. **Ensure format literals are always consumed**: Modify the assertion lowering to wrap format literals in a print operation

3. **Add cleanup pass**: Run a pass before LowerArcToLLVM that removes orphaned format strings or converts them to no-ops

4. **Mark illegal with graceful failure**: Instead of marking format ops as legal, provide a lowering pattern that emits a warning/no-op for orphaned formats

## Reproduction Command

```bash
circt-verilog --ir-hw source.sv | arcilator | opt -O0 | llc -O0 --filetype=obj -o output.o
```

## Related CIRCT Code References

- [LowerArcToLLVM.cpp - Format ops marked legal](https://github.com/llvm/circt/blob/main/lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp#L1234-L1239)
- [LowerArcToLLVM.cpp - foldFormatString](https://github.com/llvm/circt/blob/main/lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp#L805)
- [SimOps.cpp - FormatLiteralOp implementation](https://github.com/llvm/circt/blob/main/lib/Dialect/Sim/SimOps.cpp)
