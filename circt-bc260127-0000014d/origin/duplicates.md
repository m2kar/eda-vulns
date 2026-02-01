# CIRCT Duplicate Issue Check Report

**Date**: 2026-01-31  
**Repository**: llvm/circt  
**Working Directory**: /home/zhiqing/edazz/eda-vulns/circt-bc260127-0000014d/origin

## Summary

Checked for duplicate GitHub issues related to **MooreToCore string port crash** in llvm/circt repository. The analysis identified **one likely duplicate** and several related feature gaps.

## Crash Details

| Field | Value |
|-------|-------|
| **Component** | MooreToCore dialect conversion |
| **Error Type** | Assertion failure |
| **Error Message** | `Assertion 'detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed` |
| **Trigger** | Module port with SystemVerilog `string` type that cannot be converted to HW type |
| **Location** | `MooreToCore.cpp:259` in `getModulePortInfo()` |

## Search Strategy

Performed 9 GitHub issue searches across llvm/circt:
1. `string port MooreToCore`
2. `ModulePortInfo sanitizeInOut`
3. `dyn_cast non-existent value`
4. `MooreToCore type conversion`
5. `SystemVerilog string type`
6. `hw::ModulePortInfo`
7. `module port`
8. `assertion failure MooreToCore`
9. Generic string/type support queries

**Total Issues Found**: 7 relevant issues

## Detailed Results

### 🔴 HIGH PRIORITY: LIKELY DUPLICATE

**Issue #8930**: [MooreToCore] Crash with sqrt/floor
- **URL**: https://github.com/llvm/circt/issues/8930
- **State**: OPEN
- **Created**: 2025-09-09
- **Similarity Score**: 13.0/20 (LIKELY DUPLICATE)

**Match Analysis**:
- ✅ **Identical Error Signature**: Same `dyn_cast on a non-existent value` assertion
- ✅ **Same Component**: MooreToCore dialect conversion
- ✅ **Same Root Cause**: Type conversion failure passing null/invalid type to dyn_cast
- ⚠️ **Different Trigger**: sqrt/floor operations vs. string ports

**Stack Trace Match**:
```
Original crash:
  sanitizeInOut() → dyn_cast<hw::InOutType>
  getModulePortInfo() → SVModuleOpConversion

Issue #8930:
  ConversionOpConversion::matchAndRewrite() → getBitWidth()
    → dyn_cast<mlir::IntegerType>
```

**Conclusion**: The root cause is identical - both are caused by MooreToCore's inability to handle certain types, resulting in null types being passed to dyn_cast operations. The specific operations differ, but they represent the same systemic bug in type conversion error handling.

---

### 🟡 MEDIUM PRIORITY: RELATED ISSUES

**Issue #8283**: [ImportVerilog] Cannot compile forward decleared string type
- **URL**: https://github.com/llvm/circt/issues/8283
- **State**: OPEN
- **Created**: 2025-03-04
- **Similarity Score**: 4.6/20 (RELATED)

**Match Analysis**:
- ✅ String type in module ports
- ✅ MooreToCore conversion failure
- ❌ Different error: legalization failure vs. dyn_cast assertion
- **Note**: Shows the feature gap (string type conversion) but different error path

**Issue #8332**: [MooreToCore] Support for StringType from moore to llvm dialect
- **URL**: https://github.com/llvm/circt/issues/8332
- **State**: OPEN
- **Created**: 2025-03-20
- **Similarity Score**: 3.7/20 (RELATED)

**Match Analysis**:
- ✅ String type in MooreToCore
- ✅ Type conversion focus
- ❌ Focuses on LLVM dialect lowering, not module ports
- **Note**: Feature request/design discussion, not a crash report

---

### 🟢 LOW PRIORITY: TANGENTIALLY RELATED

**Issue #8382**: [FIRRTL] Crash with fstring type on port
- **Similarity Score**: 4.4/20
- **Note**: Similar pattern (string type crash on ports) but in FIRRTL dialect, not Moore

**Issue #5640**: [SV] Introduce SystemVerilog `string` type
- **Similarity Score**: 2.4/20
- **Note**: Feature request for SV dialect string type infrastructure

**Issue #8269** & **#8476**: Generic MooreToCore features
- **Similarity Score**: < 2.0/20
- **Note**: Unrelated feature requests

---

## Recommendation

| Field | Value |
|-------|-------|
| **Action** | Report as **new issue** |
| **Top Duplicate** | #8930 (Likely Duplicate) |
| **Top Score** | 13.0/20 |
| **Confidence** | HIGH |

### Reasoning

While Issue #8930 has the **identical error signature and root cause**, the current report should be submitted as a **new issue** because:

1. **Different Trigger Path**: Our crash is specifically triggered by **module ports with string types** calling `ModulePortInfo::sanitizeInOut()`, while #8930 is triggered by **conversion operations on real types** in `ConversionOpConversion`.

2. **Different Test Case**: Our minimal test case (`module test(input string a, ...)`) is simpler and more focused on the port handling aspect.

3. **Better Documentation**: This report provides more detailed analysis of the `ModulePortInfo::sanitizeInOut()` path, which is crucial for understanding the port-specific manifestation.

### Action Items

1. **Reference #8930** in the new issue as a related crash with the same root cause
2. **Consolidate** both issues under a broader fix addressing type conversion error handling in MooreToCore
3. **Add both test cases** to the fix verification
4. **Monitor #8283 and #8332** as they represent the underlying missing string type support

---

## Similarity Scoring Reference

- **0-3**: Unrelated
- **4-7**: Vague similarity
- **8-10**: Related but different
- **11-15**: Likely duplicate ✓ (Issue #8930)
- **16-20**: Strong duplicate

---

## Notes

- All search queries executed successfully
- No newer duplicates found beyond issue tracking dates
- MooreToCore dialect has multiple type conversion gaps
- SystemVerilog string type support is a known missing feature (see #5640, #8283, #8332)
