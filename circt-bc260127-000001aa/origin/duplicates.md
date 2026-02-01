# Duplicate Check Report

## Queries
- Sig2RegPass
- cannot RAUW a value with itself
- UseDefLists.h:213

## Candidates
1. #4780 [FIRRTL] Missed comb cycle, crash in canonicalizer
   - https://github.com/llvm/circt/issues/4780
   - Score: 2.0 (different dialect / pass)
2. #3600 [FIRRTL] Crash using an instance input ports to drive another instance
   - https://github.com/llvm/circt/issues/3600
   - Score: 1.5 (different dialect / pass)

## Recommendation
**likely_new** — no strong match for LLHD Sig2RegPass self-RAUW assertion.
