# Duplicate Check Report

## Summary

**Recommendation**: ⚠️ **REVIEW_EXISTING** - High-confidence duplicate found

**Top Match**: Issue [#8863](https://github.com/llvm/circt/issues/8863) (Score: 12.5/15)

---

## Crash Signature

| Field | Value |
|-------|-------|
| Assertion | `op->use_empty() && "expected 'op' to have no uses"` |
| Function | `extractConcatToConcatExtract` |
| File | `lib/Dialect/Comb/CombFolds.cpp` |
| Dialect | Comb |

---

## Duplicate Analysis

### Issue #8863: [Comb] Concat/extract canonicalizer crashes on loop

**URL**: https://github.com/llvm/circt/issues/8863  
**State**: 🟢 OPEN  
**Score**: 12.5 / 15 (Very High)

#### Match Details

| Criterion | Match | Weight | Score |
|-----------|-------|--------|-------|
| Title keywords (concat, extract, canonicalize) | ✅ | 2.0 | 2.0 |
| Body keywords (extractConcatToConcatExtract, use_empty, replaceOpAndCopyNamehint) | ✅ | 1.0 | 1.0 |
| Assertion message (`op->use_empty()`) | ✅ | 3.0 | 3.0 |
| Dialect label (Comb) | ✅ | 1.5 | 1.5 |
| Stack trace match | ✅ | 5.0 | 5.0 |
| **Total** | | | **12.5** |

#### Evidence of Duplication

**Our crash stack trace**:
```
mlir::RewriterBase::eraseOp
mlir::RewriterBase::replaceOp
circt::replaceOpAndCopyNamehint
extractConcatToConcatExtract (CombFolds.cpp)
```

**Issue #8863 stack trace**:
```
#8 mlir::RewriterBase::eraseOp(mlir::Operation*)
#9 mlir::RewriterBase::replaceOp(mlir::Operation*, mlir::ValueRange)
#10 circt::replaceOpAndCopyNamehint(...)
#11 extractConcatToConcatExtract(...) CombFolds.cpp:513:3
```

**Root Cause**: Both crashes occur when the `extractConcatToConcatExtract` canonicalization pattern attempts to replace an operation that still has uses, triggered by cyclic dependencies in extract/concat chains.

---

## Other Related Issues

### Issue #8024: [Comb] Crash in AndOp folder
- **Score**: 3.5 / 15 (Low)
- **State**: OPEN
- **Relation**: Different crash in Comb dialect canonicalization, but different root cause (AndOp fold vs extract/concat)

### Issue #8690: [Canonicalize] Non-terminating canonicalization
- **Score**: 2.5 / 15 (Low)
- **State**: CLOSED
- **Relation**: Related to canonicalizer loops, but manifests as infinite loop rather than assertion failure

### Issue #4688: comb::ExtractOp::canonicalize is very slow
- **Score**: 2.0 / 15 (Low)
- **State**: OPEN
- **Relation**: Performance issue in same canonicalization code, not a crash

---

## Recommendation

**Action**: This crash is very likely a duplicate of Issue #8863.

**Options**:
1. **Add comment to existing issue** with the new reproduction case (SystemVerilog with packed struct arrays)
2. **Wait for issue resolution** - #8863 is still open and under investigation

**If submitting as new issue**: Reference #8863 and note that this is a different trigger for the same underlying bug in `extractConcatToConcatExtract`.

---

## Search Terms Used

1. `extractConcatToConcatExtract`
2. `comb.extract canonicalize`
3. `use_empty eraseOp`
4. `comb.concat assertion`
5. `replaceOpAndCopyNamehint`
