# Duplicate Check

## Summary
No exact match found for `sim.fmt.literal` legalization failure. Two related
arcilator legalization issues exist but involve different illegal ops.

## Top Candidates
1. #9467 - `[circt-verilog][arcilator] arcilator fails to lower llhd.constant_time`
   * Similarity: 6.5/10
   * Overlap: arcilator pipeline, legalization failure
   * Difference: illegal op is `llhd.constant_time`, not `sim.fmt.literal`

2. #8286 - `[circt-verilog][llhd][arcilator] Verilog-to-LLVM lowering issues`
   * Similarity: 5/10
   * Overlap: arcilator lowering issues
   * Difference: unrelated legalization error (`llhd.constant_time`)

## Recommendation
`likely_new`
