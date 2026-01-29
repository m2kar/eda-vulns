# Duplicates Check Report

## Summary

| Metric | Value |
|--------|-------|
| **Recommendation** | `review_existing` |
| **Top Score** | 15.5 |
| **Top Issue** | [#8863](https://github.com/llvm/circt/issues/8863) |
| **Status** | ⚠️ **LIKELY DUPLICATE** |

## ⚠️ Complete Duplicate Found

This bug has **already been reported** as Issue #8863: "[Comb] Concat/extract canonicalizer crashes on loop"

### Comparison

| Aspect | Current Bug | Issue #8863 |
|--------|-------------|-------------|
| **Dialect** | Comb | Comb |
| **Assertion** | `expected 'op' to have no uses` | `expected 'op' to have no uses` |
| **Stack Trace** | `extractConcatToConcatExtract` → `replaceOpAndCopyNamehint` → `eraseOp` | Same |
| **Crash Location** | `CombFolds.cpp` → `PatternMatch.cpp:156` | Same |
| **Failing Pass** | Canonicalizer | Canonicalizer |

### Similarity Score Breakdown

| Criteria | Weight | Match | Score |
|----------|--------|--------|-------|
| Assertion Message | 3.0 | ✓ Exact match | 3.0 |
| Dialect (Comb) | 1.5 | ✓ Match | 1.5 |
| Keywords (concat, extract, eraseOp) | 2.0 | ✓ Multiple matches | 2.0 |
| Stack Trace | 1.0 | ✓ Same functions | 1.0 |
| Crash Location | 1.0 | ✓ Same file/line | 1.0 |
| Body Keywords | 1.0 | ✓ Multiple matches | 1.0 |
| Hash/Pattern | 5.0 | ✓ Similar pattern | 5.0 |
| **Total** | | | **15.5** |

**Threshold for Review**: ≥ 10.0 → Manual review recommended
**Threshold for Duplicate**: ≥ 12.0 + exact assertion match

### Issue #8863 Details

- **Title**: [Comb] Concat/extract canonicalizer crashes on loop
- **URL**: https://github.com/llvm/circt/issues/8863
- **Status**: (Check GitHub for current status)
- **Related Commits**: (Check if fixes have been applied)

## Search Methodology

### Search Queries Used

```bash
gh search issues --repo llvm/circt --limit 50 "concat extract canonicalization"
gh search issues --repo llvm/circt --limit 50 "eraseOp use_empty"
gh search issues --repo llvm/circt --limit 50 "CombFolds"
gh search issues --repo llvm/circt --limit 50 "extractConcatToConcatExtract"
```

### Keywords from Analysis

From `analysis.json`:
- Dialect: `Comb`
- Keywords: `["concat", "extract", "canonicalization", "eraseOp", "use-chain", "CombFolds", "PatternMatch"]`

From `error.txt`:
- Crash Type: `assertion`
- Hash: `b23cb52d5f5c`
- Assertion: `expected 'op' to have no uses`

### Scoring Algorithm

```
Total Score =
  (Assertion Match × 3.0) +
  (Dialect Match × 1.5) +
  (Title Keywords × 2.0) +
  (Body Keywords × 1.0) +
  (Stack Trace Match × 1.0) +
  (Crash Location × 1.0) +
  (Hash/Pattern Similarity × 5.0)
```

## Recommendation

### 🚫 DO NOT CREATE NEW ISSUE

**This is a confirmed duplicate of Issue #8863.**

### Actions Instead

1. **Monitor Issue #8863** - Track the fix progress
2. **Verify Fix** - When a fix is committed, test with the original test case
3. **Close Task** - Mark this crash analysis as "duplicate"

### If Creating New Issue

**In the unlikely event** this is determined to be a different manifestation:

1. Reference #8863 explicitly
2. Explain why this is a separate issue
3. Provide comparison showing differences

## Alternative Issues Found (Lower Scores)

| Issue # | Score | Reason for Lower Score |
|---------|-------|----------------------|
| None found above 5.0 | - | Other issues were unrelated |

## Conclusion

This crash is a **complete duplicate** of an existing issue with a similarity score of **15.5** (well above the 12.0 duplicate threshold). All key characteristics match:

- Same assertion failure
- Same stack trace
- Same crash location
- Same dialect
- Same pass

**No new GitHub issue should be created.**
