# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 12 |
| Top Similarity Score | 9.0 |
| **Recommendation** | **review_existing** |

## Search Parameters

- **Dialect**: Arc
- **Failing Pass**: LowerState
- **Crash Type**: Assertion
- **Keywords**: inout port arcilator, LLHD ref type LowerState, StateType known bit width, arc dialect inout, sequential logic inout, !llhd.ref<i1>

## Top Similar Issues

### [#9574](https://github.com/llvm/circt/issues/9574) (Score: 9.0)

**Title**: [Arc] Assertion failure when lowering inout ports in sequential logic

**State**: OPEN

**Labels**: Arc, Involving `arc` dialect

**Match Reasons**:
- ✅ Exact match on assertion context (inout ports in sequential logic)
- ✅ Same dialect (Arc)
- ✅ Same tool (arcilator)
- ✅ Keyword match on 'inout', 'arcilator', 'LowerState'
- ✅ Assertion failure type matches

**Conclusion**: This issue describes the EXACT same problem as our test case.

---

### [#4916](https://github.com/llvm/circt/issues/4916) (Score: 4.0)

**Title**: [Arc] LowerState: nested arc.state get pulled in wrong clock tree

**State**: OPEN

**Match Reasons**:
- ✅ Same dialect (Arc)
- ✅ Same failing pass (LowerState)
- ✅ Keyword match on 'LowerState'

**Conclusion**: Related issue but describes a different aspect of LowerState (nested states vs inout types).

---

## Recommendation

**Action**: `review_existing`

⚠️ **Review Required**

A highly similar issue was found. Issue #9574 describes the EXACT same problem:
- Assertion failure when lowering inout ports in sequential logic
- Same tool: arcilator
- Same dialect: Arc
- Same failing pass: LowerState

**Actions**:
1. Review issue #9574 details to confirm it's the same problem
2. If #9574 matches our issue exactly:
   - Add our test case as a comment on #9574
   - Update status.json to mark as 'duplicate'
   - Do NOT create a new GitHub issue
3. If #9574 is different:
   - Proceed to generate a new issue
   - Reference #9574 as a related issue
   - Highlight the differences

## Search Results Summary

Additional related issues found during search:
- #4532: [ExportVerilog] always_comb ordering issue
- #8013: [LLHD] Canonicalizer for processes produced by always @(*)
- #8232: [Arc] Flatten public modules
- #8484: [arcilator] Add functionality to the arcilator runtime library

These are related to Arc dialect or arcilator but describe different problems.

## Similarity Scoring

The top issue (#9574) received a score of 9.0 based on:
- Title/Body keyword match (max 2.0): 2.0
- Assertion message match (3.0): 3.0
- Dialect label match (1.5): 1.5
- Failing pass match (2.0): 2.5
  (partial credit as pass name not explicitly in title but in description)

**Total**: 9.0 / 10.0 (High similarity)

## Conclusion

This test case appears to be a DUPLICATE of issue #9574. The existing issue was created on 2026-02-01 (today) and describes the exact same problem: assertion failure when arcilator lowers inout ports in sequential logic.

**Recommended Action**: DO NOT create a new issue. Instead, add this test case as additional information to issue #9574.
