# Root Cause Analysis - 8914624e1ae9

## Executive summary
The crash is a **timeout** during a CIRCT→Arcilator→LLVM pipeline while compiling a tiny SystemVerilog module. The only observable evidence is a 60s timeout with no stack trace, which suggests a **non-terminating/very slow lowering path** in the HW-to-LLVM pipeline (most likely Arcilator or downstream LLVM passes) rather than a front-end parse failure.

## Crash context
- **Command**: `circt-verilog --ir-hw ... | arcilator | opt -O0 | llc -O0`
- **CIRCT version**: 1.139.0 (from tool path)
- **Crash type**: Timeout after 60s
- **Dialect/pipeline**: SV front-end → HW dialect → Arcilator → LLVM IR → opt/llc
- **Stack trace/assertion**: None available

## Error analysis
- Error message: `Compilation timed out after 60s`
- No assertion, stack trace, or failing pass reported.
- With such a minimal test case, a 60s timeout is anomalous and likely indicates:
  1) a non-terminating transformation/analysis loop, or
  2) accidental exponential IR growth leading to long LLVM pass time.

## Test case analysis
**Language**: SystemVerilog

**Module**:
```sv
module test_module(
  input logic clk,
  output logic r1_out
);
  logic r1;
  always_ff @(posedge clk) begin
    r1 <= ~r1;
  end
  assign r1_out = r1;
endmodule
```

**Key constructs**:
- `always_ff @(posedge clk)` sequential block
- self-referential non-blocking assignment: `r1 <= ~r1;`
- single-bit register with no reset
- continuous assign from reg to output

**Potentially problematic patterns**:
- feedback through a register with a bitwise inversion
- no initialization/reset (leaves `r1` as `X` in 4-state semantics)

## CIRCT source analysis
`../circt-src` is **not available**, so the exact crash site cannot be inspected. Analysis is based on the known pipeline and the minimal SV test case.

## Root cause hypotheses (ranked)
1. **Arcilator lowering or analysis loop fails to converge on register feedback** (Confidence: **0.45**)
   - Evidence: timeout on a trivial design suggests a non-terminating algorithm rather than heavy computation. Arcilator performs lowering and may attempt fixpoint iteration or cycle checks. Misclassifying sequential feedback as combinational could lead to an infinite loop.

2. **Arcilator emits LLVM IR that triggers pathological compile time in LLVM passes** (Confidence: **0.35**)
   - Evidence: pipeline includes `opt -O0` and `llc -O0`. A bug causing massive IR expansion (e.g., unintended unrolling or recursion) would explain a 60s timeout even on small input.

3. **Pipe/IO deadlock between stages** (Confidence: **0.20**)
   - Evidence: the toolchain is connected by Unix pipes. If any stage waits for more input or blocks on output, the pipeline may appear hung until the timeout.

## Suggested fix directions
- **Isolate the stage**: run each stage with timing and intermediate dumps (e.g., `circt-verilog --ir-hw -o out.mlir`, `arcilator out.mlir -o out.ll`) to see which stage stalls.
- **Check Arcilator handling of self-referential registers**: ensure register feedback is not treated as combinational loop in dependency analysis.
- **Inspect generated LLVM IR size**: if `out.ll` is huge, focus on Arcilator codegen for the register update logic.
- **Add watchdog/iteration caps** in fixpoint analyses to fail fast with diagnostic rather than hanging.

## Keywords for issue search
`arcilator`, `HW dialect`, `always_ff`, `self-inverting register`, `timeout`, `fixpoint`, `cycle detection`, `llvm opt hang`, `circt-verilog --ir-hw`
