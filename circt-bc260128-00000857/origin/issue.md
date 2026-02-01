## Description
Legalization failure in the `arcilator` pipeline when handling `sim.fmt.literal` operations. The crash is triggered by an immediate assertion statement with an `$error()` message containing a string literal. The `sim` dialect formatting operation is generated but has no lowering path in the arcilator flow, despite being marked as "legal" in the conversion pass.

**Crash Type**: Legalization Failure
**Dialect**: `sim`
**Failing Operation**: `sim.fmt.literal`

## Steps to Reproduce
1. Save the following code as `test.sv`:
```systemverilog
module M(input q);
  always @(*) assert(q) else $error("");
endmodule
```
2. Run the following command:
```bash
circt-verilog --ir-hw test.sv | arcilator
```

## Test Case
```systemverilog
// Minimal reproducer: sim.fmt.literal legalization failure in arcilator
// Trigger: Immediate assertion with $error() generates orphaned sim.fmt.literal
module M(input q);
  always @(*) assert(q) else $error("");
endmodule
```

## Error Output
```
error: failed to legalize operation 'sim.fmt.literal'
    %0 = sim.fmt.literal "Error: "
         ^
note: see current operation: %7 = "sim.fmt.literal"() <{literal = "Error: "}> : () -> !sim.fstring
```

## Root Cause Analysis
The `LowerArcToLLVM.cpp` pass marks `sim::FormatLiteralOp` (and other `sim.fmt.*` operations) as legal, with the expectation that they will be consumed by the lowering of `sim::PrintFormattedOp`. 

In the `arcilator` pipeline, immediate assertions with `$error()` clauses are imported into IR that includes `sim.fmt.literal` for the error message. However, the arcilator flow currently lacks a consumer for these operations when they originate from assertions. Since the operations are marked "legal" but are never actually lowered to LLVM or removed by a consumer, the MLIR legalization mechanism fails when it finds these unconverted operations remaining in the IR.

Evidence from `lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp`:
```cpp
// Mark sim::Format*Op as legal. These are not converted to LLVM, but the
// lowering of sim::PrintFormattedOp walks them to build up its format string.
target.addLegalOp<sim::FormatLiteralOp, ...>();
```

## Environment
- **CIRCT version**: firtool-1.139.0
- **LLVM version**: 22.0.0git
- **OS**: Linux
- **Architecture**: x86_64

## Related Issues
- #6810: [Arc] Add basic assertion support (Related tracking issue)
- #9467: [circt-verilog][arcilator] `arcilator` fails to lower `llhd.constant_time` (Similar legalization failure pattern)
- #7692: [Sim] Combine integer formatting ops into one op (Related dialect infrastructure)

---
### Quality Checklist
- [x] Minimized test case included
- [x] Clear reproduction steps provided
- [x] Root cause hypothesis explained
- [x] Environment details included
- [x] Related issues linked
