# CIRCT Bug Duplicate Check Report

**Date**: 2026-02-01 01:45:21  
**Test Case ID**: 260128-00000db6

## Summary

- **Dialect**: `comb`
- **Crash Type**: `assertion_failure`
- **Assertion**: `expected 'op' to have no uses`
- **Severity**: high
- **Reproducibility**: deterministic

## Root Cause

**Summary**: replaceOpAndCopyNamehint attempts to erase an operation that still has uses during ExtractOp canonicalization

**Details**: When canonicalizing extract(concat(...)) pattern with mixed assign/always_comb array assignments, the extractConcatToConcatExtract function may produce a single-element result that triggers replaceOpAndCopyNamehint. This replacement incorrectly attempts to erase an operation that is still referenced by other IR nodes.

**Pattern**: `extract(lo, cat(a, b, ...)) -> single element extraction`

**Trigger Condition**: `reverseConcatArgs.size() == 1 in extractConcatToConcatExtract`

## Crash Location

- **File**: `llvm/mlir/lib/IR/PatternMatch.cpp`
- **Function**: `mlir::RewriterBase::eraseOp`
- **Line**: 156

## Trigger Location

- **File**: `lib/Dialect/Comb/CombFolds.cpp`
- **Function**: `extractConcatToConcatExtract`
- **Line**: 548

## Duplicate Check Results

### Search Strategy

**Search Terms** (9):
```
ExtractOp, assertion_failure, canonicalization, comb, eraseOp, expected 'op' to have no uses, extractConcatToConcatExtract, replaceOpAndCopyNamehint, use_empty
```

**Search Date**: 2026-02-01T01:45:09.234787Z

### Top Findings

**Recommendation**: `REVIEW_EXISTING`  
**Highest Similarity Score**: 13.5/15  
**Top Related Issue**: #8863


## Related Issues (2)

### 1. Issue #8863: [Comb] Concat/extract canonicalizer crashes on loop

- **State**: OPEN
- **Similarity Score**: 13.5/15
- **Matched Terms** (8): 
  - eraseop, use_empty, canonicalize, extractop, comb, expected 'op' to have no uses, assertion, replaceop
- **Reason**: Matched 8 key terms in OPEN issue

---

### 2. Issue #8973: [MooreToCore] Lowering to math.ipow?

- **State**: OPEN
- **Similarity Score**: 0.0/15
- **Matched Terms** (0): 
  - None
- **Reason**: Matched 0 key terms in OPEN issue

---


## Analysis for Top Match (Issue #8863)

This is the most similar existing issue with a similarity score of 13.5/15.

### Matching Factors

The issue matches on:
1. **Same canonicalization logic** - Both involve `ExtractOp::canonicalize`
2. **Same error pattern** - Uses `replaceOpAndCopyNamehint` which calls `eraseOp`
3. **Same dialect** - Both issues are in the `comb` dialect
4. **Same assertion failure** - Both trigger `expected 'op' to have no uses`

### Recommendation

Based on the high similarity score (13.5/15), **REVIEW EXISTING ISSUE** before creating a new one:
- Check if Issue #8863 has a fix or workaround
- Verify if your test case is a duplicate or variation
- Add additional test case if it's a new variation
- Reference both issues if creating a fix


## Key Functions Involved

- `extractConcatToConcatExtract`
- `circt::comb::ExtractOp::canonicalize`
- `circt::replaceOpAndCopyNamehint`
- `mlir::RewriterBase::eraseOp`


## Files Involved

- `lib/Dialect/Comb/CombFolds.cpp`
- `lib/Support/Naming.cpp`
- `llvm/mlir/lib/IR/PatternMatch.cpp`
