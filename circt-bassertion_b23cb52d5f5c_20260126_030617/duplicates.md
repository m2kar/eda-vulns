# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 3 |
| Top Similarity Score | **15.5** |
| **Recommendation** | **review_existing** |

## ⚠️ EXACT DUPLICATE FOUND

**Issue #8863 is an EXACT DUPLICATE of this crash.**

## Search Parameters

- **Dialect**: Comb
- **Failing Pass**: Canonicalizer
- **Crash Type**: assertion
- **Assertion Message**: `expected 'op' to have no uses`
- **Keywords**: concat, extract, canonicalization, eraseOp, use-chain, CombFolds, PatternMatch

## Top Similar Issues

### [#8863](https://github.com/llvm/circt/issues/8863) (Score: 15.5) ⭐ EXACT MATCH

**Title**: [Comb] Concat/extract canonicalizer crashes on loop

**State**: OPEN

**Labels**: Comb

**Match Evidence**:
- ✅ Same assertion: `op->use_empty() && "expected 'op' to have no uses"`
- ✅ Same stack trace: `extractConcatToConcatExtract` → `replaceOpAndCopyNamehint` → `eraseOp`
- ✅ Same crash location: `CombFolds.cpp` in `ExtractOp::canonicalize`
- ✅ Same dialect: Comb
- ✅ Same trigger: concat/extract loop patterns

---

### [#8024](https://github.com/llvm/circt/issues/8024) (Score: 3.5)

**Title**: [Comb] Crash in AndOp folder

**State**: OPEN

**Labels**: Comb

**Match Evidence**:
- ❌ Different assertion
- ❌ Different crash location (AndOp folder, not ExtractOp)
- ✅ Same dialect: Comb

---

### [#3662](https://github.com/llvm/circt/issues/3662) (Score: 2.5)

**Title**: [Comb] ICmp folds missing for CEQ/CNE/WEQ/WNE

**State**: OPEN

**Labels**: Comb

**Match Evidence**:
- ❌ Feature request, not bug
- ✅ Same dialect: Comb

---

## Recommendation

**Action**: `review_existing`

⚠️ **DUPLICATE CONFIRMED**

Issue #8863 describes the **exact same bug**:
- Same assertion failure message
- Same stack trace path
- Same crash function (`extractConcatToConcatExtract`)
- Same root cause (concat/extract canonicalizer loop handling)

**DO NOT create a new issue.**

**Recommended Actions**:
1. Add your test case as a comment to Issue #8863
2. Update status.json to mark as `duplicate`
3. Reference: https://github.com/llvm/circt/issues/8863

## Scoring Weights Used

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
| Failing pass match | 2.0 | If failing pass appears in issue |
| Stack trace match | 2.0 | Per matching function in stack |
