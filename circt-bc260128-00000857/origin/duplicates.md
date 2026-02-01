# CIRCT Duplicate Issue Check Report

**Crash ID**: circt-bc260128-00000857  
**Generated**: 2026-01-31T00:00:00Z  
**Status**: LIKELY NEW ISSUE

## Summary

A comprehensive search of the CIRCT GitHub repository was performed to identify potential duplicate issues. The search covered multiple key terms related to this crash:
- sim.fmt.literal operation failures
- Legalization errors in sim dialect
- Assertion handling in Arc pipeline
- Arcilator lowering failures

**Result**: No exact duplicates found. However, three related issues were identified as having some relevance.

## Search Strategy

The following search queries were executed against the llvm/circt repository:

1. `sim.fmt.literal` - Direct operation name
2. `failed to legalize sim` - Legalization failures in sim dialect
3. `assertion error message` - Assertion handling errors
4. `sim dialect` - Sim dialect specific issues
5. `error message string` - Error message formatting
6. `assertion failed` - General assertion failures
7. `Arc arcilator` - Arc and arcilator pipeline issues
8. `sim.fmt` - Sim format operations family

## Top Similar Issues

### Issue #6810: [Arc] Add basic assertion support
- **URL**: https://github.com/llvm/circt/issues/6810
- **State**: OPEN
- **Similarity Score**: 7/15
- **Labels**: good first issue, Arc
- **Created**: 2024-03-15T01:45:57Z

**Relevance Analysis**:
- ✓ Matches crash type: assertion
- ✓ Matches dialect: Arc
- ✓ Related to assertion infrastructure in Arc

**Key Differences**:
- Issue #6810 is a high-level tracking issue for adding `verif.assert` and `sv.assert.concurrent` support
- This crash is about a low-level sim dialect formatting operation failure
- The crash occurs during legalization in the arcilator pipeline for sim.fmt.literal ops
- Issue #6810 doesn't mention sim.fmt.literal or the specific formatting ops

**Verdict**: Related but NOT a duplicate - this crash is a symptom of missing infrastructure referenced in #6810

---

### Issue #9467: [circt-verilog][arcilator] `arcilator` fails to lower `llhd.constant_time` generated from simple SV delay (`#1`)
- **URL**: https://github.com/llvm/circt/issues/9467
- **State**: OPEN
- **Similarity Score**: 6/15
- **Labels**: LLHD, Arc
- **Created**: 2026-01-20T17:10:39Z

**Relevance Analysis**:
- ✓ Matches crash type: legalization failure
- ✓ Matches tool: arcilator pipeline
- ✓ Similar pattern: operation marked as illegal with no lowering path

**Key Differences**:
- Issue #9467 is about `llhd.constant_time` operation
- This crash is about `sim.fmt.literal` operation
- Different dialects involved (LLHD vs Sim)
- Different conversion pipelines failing

**Verdict**: Related pattern but NOT a duplicate - different operation and dialect

---

### Issue #7692: [Sim] Combine integer formatting ops into one op
- **URL**: https://github.com/llvm/circt/issues/7692
- **State**: OPEN
- **Similarity Score**: 4/15
- **Labels**: Simulator
- **Created**: 2024-10-10T20:39:13Z

**Relevance Analysis**:
- ✓ Matches dialect: sim
- ✓ Related to sim formatting operations (sim.fmt family)
- ✓ Addresses sim dialect infrastructure

**Key Differences**:
- Issue #7692 is about consolidating integer formatting ops (hex, bin, dec)
- This crash is specifically about `sim.fmt.literal` (string formatting)
- Issue #7692 is a feature request/enhancement, not a bug
- Different formatting operation family (integer vs. string)

**Verdict**: Related infrastructure but NOT a duplicate - different ops and different issue type

---

## Detailed Similarity Scoring

### Scoring Criteria
- Matching crash operation: +5 points
- Matching dialect: +3 points
- Matching crash type: +3 points
- Similar error message: +2 points
- Matching test case features: +2 points

### Issue #6810 Score Breakdown
- Matching crash operation: 0 (no sim.fmt.literal mentioned)
- Matching dialect: 3 (Arc/assertion handling)
- Matching crash type: 3 (assertion)
- Similar error message: 1 (general assertion context)
- Matching test case features: 0
- **Total: 7/15**

### Issue #9467 Score Breakdown
- Matching crash operation: 0 (llhd.constant_time, not sim.fmt.literal)
- Matching dialect: 0 (LLHD, not Sim)
- Matching crash type: 3 (legalization failure)
- Similar error message: 2 ("failed to legalize" pattern)
- Matching test case features: 1 (arcilator pipeline)
- **Total: 6/15**

### Issue #7692 Score Breakdown
- Matching crash operation: 0 (sim.fmt.int/hex/bin/dec, not sim.fmt.literal)
- Matching dialect: 3 (Sim dialect)
- Matching crash type: 0 (not a crash)
- Similar error message: 0 (feature request)
- Matching test case features: 1 (sim formatting operations)
- **Total: 4/15**

---

## Conclusion

### Recommendation: **LIKELY NEW ISSUE**

This crash represents a new, previously unreported issue for the following reasons:

1. **Specific Operation**: The crash involves `sim.fmt.literal` operation legalization failure, which is not mentioned in any of the three related issues found.

2. **Unique Root Cause**: The crash occurs because immediate assertions with `$error()` messages generate orphaned `sim.fmt.literal` operations that have no consumer in the arcilator flow. This is a specific missing lowering pattern.

3. **Specific Pipeline Context**: The crash happens during the arcilator legalization pipeline when lowering sim operations to LLVM, which is distinct from the parent assertion support tracking issue (#6810).

4. **Implementation Gap**: The root cause indicates that `LowerArcToLLVM.cpp` marks `sim::FormatLiteralOp` as legal expecting it to be consumed by `PrintFormattedOp`, but orphaned ops are not handled.

### Recommended Action
- **Create a new GitHub issue** to report this specific crash
- **Reference Issue #6810** as related context (parent assertion support tracking)
- **Provide minimal test case** showing sim.fmt.literal legalization failure
- **Include root cause analysis** from this report

---

## Search Results Summary

| Query | Results Found | Most Relevant |
|-------|---|---|
| sim.fmt.literal | 0 direct matches | N/A |
| failed to legalize sim | 2 issues | #9467 |
| assertion error message | 3 issues | #6810 |
| sim dialect | 13 issues | #7692 |
| error message string | 3 issues | #7531 |
| assertion failed | 47 issues | #6810 |
| Arc arcilator | 11 issues | #6810 |
| sim.fmt | 0 direct matches | #7692 |

**Total issues searched**: 50+ issues across all queries  
**Exact duplicates found**: 0  
**Potential parent issues**: 1 (#6810 - assertion support tracking)  
**Related infrastructure issues**: 2 (#9467, #7692)

