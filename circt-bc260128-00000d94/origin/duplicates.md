# Duplicate Check Report

**Case ID**: 260128-00000d94  
**Timestamp**: 2026-02-01T01:15:16.843367

## Crash Signature

| Property | Value |
|----------|-------|
| Assertion | dyn_cast on a non-existent value |
| Pass | convert-moore-to-core |
| Dialect | moore |
| File | lib/Conversion/MooreToCore/MooreToCore.cpp |
| Function | getModulePortInfo |

## Search Results

**Total Issues Found**: 22  
**Top Similarity Score**: 13.5/20

## Top 10 Similar Issues

### 1. #8930 (Score: 13.5) [OPEN]

**Title**: [MooreToCore] Crash with sqrt/floor

**URL**: https://github.com/llvm/circt/issues/8930

---

### 2. #8332 (Score: 9.5) [OPEN]

**Title**: [MooreToCore] Support for StringType from moore to llvm dialect

**URL**: https://github.com/llvm/circt/issues/8332

---

### 3. #8283 (Score: 7.0) [OPEN]

**Title**: [ImportVerilog] Cannot compile forward decleared string type

**URL**: https://github.com/llvm/circt/issues/8283

---

### 4. #7629 (Score: 6.5) [OPEN]

**Title**: [MooreToCore] Support net op

**URL**: https://github.com/llvm/circt/issues/7629

---

### 5. #8163 (Score: 6.5) [OPEN]

**Title**: [MooreToCore] Out-of-bounds moore.extract lowered incorrectly

**URL**: https://github.com/llvm/circt/issues/8163

---

### 6. #8269 (Score: 6.5) [OPEN]

**Title**: [MooreToCore] Support `real` constants

**URL**: https://github.com/llvm/circt/issues/8269

---

### 7. #8476 (Score: 6.5) [OPEN]

**Title**: [MooreToCore] Lower exponentiation to `math.ipowi`

**URL**: https://github.com/llvm/circt/issues/8476

---

### 8. #8973 (Score: 6.5) [OPEN]

**Title**: [MooreToCore] Lowering to math.ipow?

**URL**: https://github.com/llvm/circt/issues/8973

---

### 9. #7918 (Score: 6.0) [OPEN]

**Title**: [MooreToCore] `moore.extract` is converted to `hw.array_slice` when array's element type is also array

**URL**: https://github.com/llvm/circt/issues/7918

---

### 10. #8276 (Score: 6.0) [OPEN]

**Title**: [MooreToCore] Support for UnpackedArrayType emission

**URL**: https://github.com/llvm/circt/issues/8276

---

## Recommendation

**Action**: `review_existing`

**Confidence**: high

**Reason**: High similarity score (>10) indicates this may be related to existing issues


### ⚠️ Review Required

A highly similar issue (#8332 or #8283) was found. This appears to be related to **missing StringType support in MooreToCore conversion**.

**Key Findings**:
- Issue #8930: Same assertion pattern but different location (sqrt/floor vs string port)
- Issue #8332: Feature request for StringType Moore->LLVM lowering  
- Issue #8283: String variable compilation failure in MooreToCore

**Recommended Next Steps**:
1. Review issue #8332 and #8283 to check if they already cover string port conversion
2. If this is a distinct manifestation (string OUTPUT port vs string variable), proceed with new report
3. If already covered, consider marking as duplicate with cross-reference
4. Otherwise, create new issue referencing related issues

## Detailed Issue Analysis

### Issue #8930: [MooreToCore] Crash with sqrt/floor

**Relationship**: RELATED but DISTINCT

**Similarity**: Same dyn_cast assertion failure  
**Difference**: Crash in sqrt/floor conversion (real type) vs string port handling  
**Verdict**: Different root cause - keep separate

---

### Issue #8332: [MooreToCore] Support for StringType from moore to llvm dialect

**Relationship**: HIGHLY RELEVANT - Feature Request

**Description**: Discussion about adding StringType support from Moore to LLVM dialect  
**Status**: Open feature request, not a crash report  
**Connection**: Addresses the broader issue of StringType handling

---

### Issue #8283: [ImportVerilog] Cannot compile forward decleared string type

**Relationship**: HIGHLY RELEVANT - Same Root Issue

**Description**: String variable (moore.variable with string type) cannot be legalized in MooreToCore  
**Status**: Open bug  
**Connection**: Directly related to StringType conversion failure in MooreToCore

---

## Scoring Weights Used

| Factor | Weight | Match |
|--------|--------|-------|
| Title keyword match | 2.0 | Per keyword found |
| Body keyword match | 1.0 | Per keyword found |
| Assertion message match | 3.0 | If present |
| Dialect label match | 1.5 | If matches |
| Failing pass match | 2.0 | If mentions pass |
| MooreToCore specific | 2.0 | If mentions module |
| Type conversion | 1.5 | If mentions conversion |
| Port/module related | 1.0 | If port-related |
| String type | 1.5 | If string mentioned |
| Dyn_cast/assertion | 1.0 | If crash-related |

## Summary

This crash is part of a **broader StringType handling limitation in MooreToCore conversion**. Multiple related issues exist but they appear to address different manifestations:

- String variables in procedures (Issue #8283)
- String type to LLVM lowering strategy (Issue #8332)
- Real type conversion assertion (Issue #8930)
- **This case**: String output port in module signature

Consider this a **new but related issue** that should be reported with cross-references to existing issues.
