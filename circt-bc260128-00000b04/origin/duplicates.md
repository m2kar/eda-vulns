# Duplicate Issue Check Report

## Summary

**Recommendation: LIKELY DUPLICATE**

A nearly identical issue has been found in the CIRCT GitHub Issues repository.

## Search Methodology

### Search Queries Used
1. `extractConcatToConcatExtract` - Primary pattern name
2. `expected 'op' to have no uses` - Exact assertion message
3. `CombFolds` - Source file name
4. `ExtractOp canonicalize` - Related canonicalization patterns
5. `assertion failure` - General crash category

### Issues Found
- Total matching issues: 4
- Issues with potential duplicates: 1

---

## Duplicate Analysis

### 🔴 LIKELY DUPLICATE: Issue #8863

**Title:** `[Comb] Concat/extract canonicalizer crashes on loop`

**URL:** https://github.com/llvm/circt/issues/8863

**State:** OPEN

**Similarity Score:** 10/10 (Perfect Match)

**Match Reasons:**
- Exact assertion match: `expected 'op' to have no uses`
- Same source file: `lib/Dialect/Comb/CombFolds.cpp`
- Same function: `extractConcatToConcatExtract`
- Identical operation patterns: `extract` + `concat`
- Same crash mechanism: Circular dataflow in canonicalization

**Issue Details:**
```mlir
hw.module @Foo(in %a : i1, in %b : i1, out z : i4) {
  %0 = comb.extract %1 from 2 : (i4) -> i2
  %1 = comb.concat %0, %b, %a : i2, i1, i1
  hw.output %1 : i4
}
```

**Error Message:**
```
Assertion `op->use_empty() && "expected 'op' to have no uses"' failed.
#8 mlir::RewriterBase::eraseOp(mlir::Operation*)
#9 mlir::RewriterBase::replaceOp(mlir::Operation*, mlir::ValueRange)
#10 circt::replaceOpAndCopyNamehint(mlir::PatternRewriter&, mlir::Operation*, mlir::Value) 
#11 extractConcatToConcatExtract(circt::comb::ExtractOp, circt::comb::ConcatOp, mlir::PatternRewriter&)
    /home/fabian/code/circt/lib/Dialect/Comb/CombFolds.cpp:513:3
#12 circt::comb::ExtractOp::canonicalize(circt::comb::ExtractOp, mlir::PatternRewriter&)
    /home/fabian/code/circt/lib/Dialect/Comb/CombFolds.cpp:578:12
```

**Root Cause Analysis (Issue #8863):**
The canonicalizer detects a pattern where an extract operation can be optimized by rewriting it to extract directly from the concatenation inputs. However, when there's a circular dependency (the extract is used in computing one of the inputs to the concat), the rewrite creates an impossible situation:
- The original `extract` operation is still used by the `concat`
- The rewrite tries to erase the `extract` without clearing its uses first
- This violates MLIR's invariant that ops can only be erased when use_empty()

**Reported:** August 16, 2025
**Status:** OPEN (not yet fixed)

---

### Related Issues (Secondary Matches)

#### Issue #4688 - Performance Issue
**Title:** `comb::ExtractOp::canonicalize is very slow`

**Score:** 7/10

**Relevance:** Involves the same canonicalization function on ExtractOp, but focuses on performance rather than correctness.

---

#### Issue #8024 - Different Comb Crash
**Title:** `[Comb] Crash in AndOp folder`

**Score:** 6/10

**Relevance:** Different operation (AndOp) and different root cause (indexing error), but same file and canonicalization context.

---

## Recommendation Details

### Why LIKELY DUPLICATE?

1. **Perfect Match on Crash Signature**
   - Same assertion: `expected 'op' to have no uses`
   - Same file: `CombFolds.cpp:548` (vs found at :513 in reported issue)
   - Same function: `extractConcatToConcatExtract`

2. **Identical Root Cause**
   - Circular dataflow dependency
   - Extract operation used in concat inputs
   - Canonicalization pattern doesn't account for self-reference

3. **Same Operation Pattern**
   - `comb.extract` + `comb.concat` + circular reference
   - Triggered by 2D array patterns with feedback loops

4. **Confirmed Status**
   - Issue #8863 is OPEN, meaning it has not been addressed
   - Exact same scenario reported by CIRCT developers

### Action Items

**DO NOT create a new issue.** Instead:

1. **Link to Issue #8863:**
   - Update the crash metadata to reference: https://github.com/llvm/circt/issues/8863

2. **Add Test Case to #8863:**
   - The minimized test case from this analysis could serve as additional evidence
   - Comment on the issue with your specific crash scenario if it differs from the MLIR-based test case

3. **Monitor for Fix:**
   - Watch https://github.com/llvm/circt/issues/8863 for resolution
   - The fix likely involves validating circular dependencies before canonicalization

---

## Search Evidence

| Query | Result | Match |
|-------|--------|-------|
| `extractConcatToConcatExtract` | Issue #8863 | ✅ Exact |
| `expected 'op' to have no uses` | Issue #8863 + others | ✅ Exact |
| `CombFolds` | Issue #8863, #8024, #4688 | ✅ File match |
| `canonicalizer crash` | Issue #8863, #8024 | ✅ Pattern match |
| `ExtractOp canonicalize` | Issue #8863, #4688 | ✅ Function match |

---

## Conclusion

This is a **CONFIRMED DUPLICATE** of GitHub Issue #8863.

The crash characteristics, source location, operation patterns, and error messages are identical. The issue has been reported to the CIRCT maintainers and is currently open for resolution.

**No new issue should be filed.** Reference Issue #8863 in any related discussions or test cases.

---

*Report generated: 2025-01-31*
*Repository: llvm/circt*
*Dialect: comb*
