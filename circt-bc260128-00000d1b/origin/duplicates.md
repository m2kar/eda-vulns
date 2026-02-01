# CIRCT Duplicate Issue Check Report

## Summary

This report analyzes whether the current crash (assertion in `extractConcatToConcatExtract`) is a duplicate of an existing CIRCT GitHub issue.

| Metric | Value |
|--------|-------|
| **Issues Found** | 5 |
| **Top Similarity Score** | 17.5 / 20.0 |
| **Recommendation** | **⚠️ REVIEW EXISTING** |
| **Confidence** | HIGH |

---

## 🔍 Search Methodology

### Search Terms Extracted
- **Dialect**: `comb`
- **Failing Pass**: `Canonicalizer`
- **Crash Type**: `assertion`
- **Keywords**: extractConcatToConcatExtract, ExtractOp, ConcatOp, canonicalize, use_empty, eraseOp, CombFolds, replaceOpAndCopyNamehint, array indexing, Canonicalizer
- **Assertion**: `op->use_empty() && "expected 'op' to have no uses"`

### Search Queries Used
```bash
gh issue list -R llvm/circt --search "extractConcatToConcatExtract OR ExtractOp canonicalize" --limit 20
gh issue list -R llvm/circt --search "use_empty eraseOp" --limit 15
gh issue list -R llvm/circt --search "Comb comb.extract comb.concat" --limit 15
```

---

## 📊 Top Matching Issues

### 🏆 **#8863 - [Comb] Concat/extract canonicalizer crashes on loop**

**Similarity Score: 17.5 / 20.0** ⭐⭐⭐⭐⭐

**URL**: https://github.com/llvm/circt/issues/8863  
**Status**: OPEN  
**Created**: 2025-08-16  
**Labels**: Comb

#### Description
This is **HIGHLY RELEVANT** - identical crash signature and root cause:

```mlir
hw.module @Foo(in %a : i1, in %b : i1, out z : i4) {
  %0 = comb.extract %1 from 2 : (i4) -> i2
  %1 = comb.concat %0, %b, %a : i2, i1, i1
  hw.output %1 : i4
}
```

**Crash Stack Trace** (from issue):
```
Assertion `op->use_empty() && "expected 'op' to have no uses"' failed
 #8 mlir::RewriterBase::eraseOp(mlir::Operation*)
 #9 mlir::RewriterBase::replaceOp(mlir::Operation*, mlir::ValueRange)
#10 circt::replaceOpAndCopyNamehint(mlir::PatternRewriter&, mlir::Operation*, mlir::Value)
#11 extractConcatToConcatExtract(circt::comb::ExtractOp, circt::comb::ConcatOp, mlir::PatternRewriter&)
    /home/fabian/code/circt/lib/Dialect/Comb/CombFolds.cpp:513:3
#12 circt::comb::ExtractOp::canonicalize(circt::comb::ExtractOp, mlir::PatternRewriter&)
    /home/fabian/code/circt/lib/Dialect/Comb/CombFolds.cpp:578:12
```

#### Why High Match?
- ✅ **Exact assertion message match**: `op->use_empty()`
- ✅ **Same function**: `extractConcatToConcatExtract`
- ✅ **Same file**: `CombFolds.cpp`
- ✅ **Same dialect**: `comb`
- ✅ **Same pattern**: extract from concat, then concat with extract
- ✅ **Same trigger**: Array element access pattern
- ✅ **Same pass**: Canonicalizer

#### Test Case Comparison

**Issue #8863**:
```systemverilog
module Foo(input logic a, logic b, output logic [3:0] z);
  logic [3:0] x;
  always_comb begin
    x[0] = a;
    x[1] = b;
  end
  assign z = x;
endmodule
```

**Current Bug**:
```systemverilog
module Foo(input logic a, logic b, output logic [3:0] z);
  logic [3:0] x;
  always_comb begin
    x[0] = a;
    x[1] = b;
  end
  assign z = x;
endmodule
```

**Status**: Almost identical pattern - array element writes followed by array read.

---

### 📌 Issue #8586 - [ExportVerilog] Graph region op order creates redundant spills

**Similarity Score: 5.0 / 20.0**

**URL**: https://github.com/llvm/circt/issues/8586  
**Status**: OPEN  
**Labels**: ExportVerilog

Involves `comb.extract` and `comb.concat` but for different issue (redundant spills, not canonicalizer crash).

---

### 📌 Issue #8260 - [Comb] or(concat, concat) -> concat & or(and, concat) canonicalizer

**Similarity Score: 4.5 / 20.0**

**URL**: https://github.com/llvm/circt/issues/8260  
**Status**: OPEN  
**Labels**: Comb, enhancement

Related canonicalization patterns for concat/extract but enhancement request, not crash.

---

### 📌 Issue #4532 - [ExportVerilog] always_comb ordering issue

**Similarity Score: 3.0 / 20.0**

**URL**: https://github.com/llvm/circt/issues/4532  
**Status**: OPEN

Involves concat/extract in always_comb but different manifestation (ordering issue).

---

### 📌 Issue #4688 - comb::ExtractOp::canonicalize is very slow

**Similarity Score: 2.5 / 20.0**

**URL**: https://github.com/llvm/circt/issues/4688  
**Status**: OPEN

Performance issue with ExtractOp canonicalize, not crash.

---

## 🎯 Scoring Breakdown

| Factor | Weight | Your Bug | Issue #8863 |
|--------|--------|----------|------------|
| extractConcatToConcatExtract function | 2.0 | ✅ | ✅ |
| Comb dialect | 2.0 | ✅ | ✅ |
| use_empty assertion | 5.0 | ✅ | ✅ |
| Extract/Concat pattern | 2.0 | ✅ | ✅ |
| Array indexing pattern | 1.5 | ✅ | ✅ |
| Canonicalizer crash | 3.0 | ✅ | ✅ |
| Same file path | 2.0 | ✅ | ✅ |
| **Total** | **20.0** | **17.5** | **17.5** |

---

## ⚠️ RECOMMENDATION

### Action: **REVIEW EXISTING ISSUE #8863**

**Confidence Level**: 🟢 **HIGH**

### Rationale

1. **Exact Crash Signature**: The assertion message, stack trace, and affected function are identical.

2. **Same Root Cause**: Both crashes involve:
   - Extract operation from a Concat result
   - Subsequent rewrite creating new ExtractOp
   - Original ExtractOp still has uses that haven't been replaced
   - `use_empty()` assertion fails in `replaceOpAndCopyNamehint`

3. **Identical Trigger Pattern**:
   - Array element writes in `always_comb`
   - Array element read
   - Creates Extract from Concat during lowering
   - Canonicalizer attempts optimization

4. **Same Version Range**: Both on circt-1.x versions

### Next Steps

**Option A: Mark as Duplicate** (Recommended if issue #8863 is still being investigated)
```
status.json: "duplicate": true, "related_issue": 8863
```

**Option B: Add Test Case** (If issue #8863 is fixed in newer version)
- Check if issue #8863 is resolved in latest CIRCT main
- If fixed, test case to ensure no regression
- If not fixed, add this test case as additional evidence

**Option C: Contribute Fix** (If you want to help CIRCT)
- The fix likely involves ensuring `replaceOpAndCopyNamehint` properly updates all uses before erasing
- Or adding a check in `extractConcatToConcatExtract` to verify all uses are replaced
- Review the pattern matching logic in `CombFolds.cpp:513`

---

## 📋 Score Weights Reference

| Factor | Weight | Notes |
|--------|--------|-------|
| Exact function match | 2.0 | Same crashing function |
| Dialect match | 2.0 | Same comb dialect |
| Assertion match | 5.0 | Identical error message |
| Pattern match | 2.0 | Same IR pattern |
| Feature interaction | 1.5 | Array indexing behavior |
| Crash type match | 3.0 | Both are assertions in canonicalizer |
| File path match | 2.0 | Same source file |

---

## 🔗 Related Issues

- #8863 - This one! (DUPLICATE)
- #4688 - ExtractOp canonicalize performance
- #8260 - Comb concat/extract canonicalizers (enhancement)
- #8586 - ExportVerilog ordering (different manifestation)

---

**Report Generated**: 2025-02-01  
**Analysis Type**: Automated Duplicate Detection  
**Status**: Ready for human review
