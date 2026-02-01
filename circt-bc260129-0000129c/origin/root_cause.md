# Root Cause Analysis

## Summary
The crash occurs during legalization after `circt-verilog` emits `sim.fmt.literal` for
`$error(...)`. The Arc/Sim lowering pipeline reaches the Arc-to-LLVM conversion without
eliminating or legalizing `sim.fmt.literal`, so the conversion target rejects the op and
fails with `failed to legalize operation 'sim.fmt.literal'`.

## Evidence

* The reproduction shows the failure at legalization time:
  * `failed to legalize operation 'sim.fmt.literal'` with the literal
    `"Error: Assertion failed: q != 0"`.

* `sim.fmt.literal` is defined as a Sim dialect format-string fragment op:
  * `include/circt/Dialect/Sim/SimOps.td` defines `FormatLiteralOp : SimOp<"fmt.literal">`.

* The Arc-to-LLVM conversion pass does **not** legalize or lower Sim format ops:
  * `lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp` builds the conversion target, but
    does not mark `sim::FormatLiteralOp` (or other `sim::Format*` ops) as legal nor
    provide conversion patterns for them.

This means any `sim.fmt.literal` left in the IR when the Arc-to-LLVM conversion runs
will cause a legalization failure, which is exactly what happens here.

## Suspected Fix Direction

* Ensure format-string ops are eliminated or lowered before Arc-to-LLVM conversion.
  Possible options:
  * Add/enable a pass that lowers `sim.fmt.*` ops to a representation consumed by
    Arc lowering (or to LLVM-compatible string constants) earlier in the pipeline.
  * Alternatively, mark `sim::Format*` ops legal in `LowerArcToLLVM` and ensure any
    users are handled, similar to how some conversions treat them as legal for
    later elimination.
