# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 2 |
| Top Similarity Score | **9.5** |
| **Recommendation** | **review_existing** |

## ⚠️ Duplicate Found

**This bug is a duplicate of [Issue #6343](https://github.com/llvm/circt/issues/6343)**

| Field | Value |
|-------|-------|
| Issue # | 6343 |
| Title | MLIR lowering issue |
| Status | OPEN |
| Similarity | 95% |

## Search Parameters

- **Dialect**: Calyx
- **Failing Pass**: lower-scf-to-calyx
- **Crash Type**: segmentation_fault
- **Keywords**: SCFToCalyx, buildCFGControl, func.call, SuccessorRange, segfault, llvm.smax

## Top Similar Issues

### [#6343](https://github.com/llvm/circt/issues/6343) (Score: 9.5) ⭐ DUPLICATE

**Title**: MLIR lowering issue

**State**: OPEN

**Match Reasons**:
- ✅ Identical crash: SCFToCalyx segfault with `func.call @llvm.smax.i32`
- ✅ Same processing pipeline: `mlir-opt --lower-affine | --scf-for-to-while | circt-opt --lower-scf-to-calyx`
- ✅ Same assertion: `!NodePtr->isKnownSentinel()`
- ✅ Same root cause: func.call in loop body not supported

**Issue Description Summary**:
> User reports MLIR implementing fixed-point 4-layer convolution crashes when lowering with `--lower-scf-to-calyx`. The code uses `func.call @llvm.smax.i32` (RELU operation) inside affine loops. Maintainer has acknowledged this is a bug in scf-to-calyx.

---

### [#5031](https://github.com/llvm/circt/issues/5031) (Score: 2.0)

**Title**: [scf-to-calyx] Mark `cf` operations illegal

**State**: OPEN

**Match Reasons**:
- Related to scf-to-calyx pass
- Different issue type (feature request for cf operations)

---

## Recommendation

**Action**: `review_existing`

⚠️ **Review Required**

A highly similar issue was found (Score: 9.5). This is confirmed to be a **DUPLICATE** of Issue #6343.

**DO NOT create a new issue.**

**Recommended Actions**:
1. Add this test case as a comment to Issue #6343 (if it provides additional value)
2. Update local tracking to mark as duplicate
3. Monitor Issue #6343 for fix progress

## Evidence Comparison

| Aspect | This Bug | Issue #6343 |
|--------|----------|-------------|
| Crash Location | `buildCFGControl` | `buildCFGControl` |
| Failing Pass | `lower-scf-to-calyx` | `lower-scf-to-calyx` |
| Trigger | `func.call @llvm.smax.i32` in loop | `func.call @llvm.smax.i32` in loop |
| Assertion | `!NodePtr->isKnownSentinel()` | `!NodePtr->isKnownSentinel()` |
| Input Dialect | affine | affine |

## Scoring Weights Used

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
| Failing pass match | 2.0 | If failing pass appears in issue |
| Crash pattern match | 3.0 | If crash signature matches |
