# CIRCT GitHub Duplicate Issue Report

## Executive Summary

**Search Date**: 2026-01-31T17:36:02.498109Z
**Repository**: llvm/circt
**Recommendation**: **NEW ISSUE**

### Key Findings
- **Top Match Score**: 5/15
- **Total Related Issues Found**: 10
- **Crash Type**: Assertion failure (dyn_cast on non-existent value)
- **Affected Component**: Moore-to-Core type conversion, ModulePortInfo::sanitizeInOut

---

## Crash Analysis

### Original Issue Details
- **Location**: include/circt/Dialect/HW/PortImplementation.h:177 (ModulePortInfo::sanitizeInOut)
- **Called From**: lib/Conversion/MooreToCore/MooreToCore.cpp:259
- **Failed Assertion**: `detail::isPresent(Val) && "dyn_cast on a non-existent value"`
- **Root Cause**: Moore-to-Core type conversion produces null mlir::Type for SV string input port
- **Affected Code**:
  - lib/Conversion/MooreToCore/MooreToCore.cpp:getModulePortInfo
  - lib/Conversion/MooreToCore/MooreToCore.cpp:SVModuleOpConversion::matchAndRewrite
  - include/circt/Dialect/HW/PortImplementation.h:ModulePortInfo::sanitizeInOut
  - Moore-to-Core TypeConverter (string type handling)

### Search Keywords Used
- `dyn_cast on a non-existent value`
- `ModulePortInfo::sanitizeInOut`
- `MooreToCore.cpp:259`
- `SVModuleOpConversion`
- `string port`
- `moore string type`
- `InOutType`
- `circt-verilog --ir-hw`
- `MooreToCorePass`

---

## Similar Issues Found

### Top 5 Most Similar Issues


#### 1. Issue #8219: [ESI]Assersion: dyn_cast on a non-existent value
- **State**: CLOSED
- **Similarity Score**: 5/15
- **Matched Keywords**: `dyn_cast on a non-existent value`
- **Link**: https://github.com/llvm/circt/issues/8219


#### 2. Issue #8939: [MooreToCore] Generating invalid llhd.sig.array_slice
- **State**: CLOSED
- **Similarity Score**: 3/15
- **Matched Keywords**: `MooreToCore`
- **Link**: https://github.com/llvm/circt/issues/8939


#### 3. Issue #8934: [MooreToCore] Unsupported $fatal to sim.terminate conversion
- **State**: CLOSED
- **Similarity Score**: 3/15
- **Matched Keywords**: `MooreToCore`
- **Link**: https://github.com/llvm/circt/issues/8934


#### 4. Issue #7627: [MooreToCore] Unpacked array causes crash
- **State**: CLOSED
- **Similarity Score**: 3/15
- **Matched Keywords**: `MooreToCore`
- **Link**: https://github.com/llvm/circt/issues/7627


#### 5. Issue #7628: [MooreToCore] Support string constants
- **State**: CLOSED
- **Similarity Score**: 3/15
- **Matched Keywords**: `MooreToCore`, `string`
- **Link**: https://github.com/llvm/circt/issues/7628


---

## Scoring Methodology

Similarity scores (0-15) are calculated as follows:
- **+5**: Exact assertion message match ("dyn_cast on a non-existent value")
- **+3**: Same component (ModulePortInfo::sanitizeInOut)
- **+3**: Same code location (MooreToCore.cpp:259)
- **+1 per keyword**: Matched search keywords (max +4)

### Score Interpretation
- **Score ≥ 10**: Likely duplicate → **REVIEW EXISTING**
- **Score 6-9**: Possibly related → **LIKELY NEW**
- **Score < 6**: Different issue type → **NEW ISSUE**

---

## Recommendation: NEW ISSUE

### Rationale

The current crash involves a **Moore-to-Core type conversion** producing a null type for SV string input ports, which then triggers an assertion in `ModulePortInfo::sanitizeInOut` when trying to cast the null value.

**Most Similar Issue**: Issue #8219 (Score: 5/15)
- This ESI dyn_cast issue shares the same assertion pattern but occurs in a different component (ESI bundle handling)
- The root cause differs: ESI unwrap/wrap handling vs Moore string type conversion
- However, it represents the same symptom: unsafe dyn_cast on potentially null values

### Key Differences from Existing Issues

1. **Specific to Moore dialect**: Most MooreToCore issues focus on unsupported operations (arrays, constants, asserts)
2. **Type conversion failure**: This is specifically about null type propagation, not operation support
3. **String port handling**: Unique to string type inputs in Moore imports
4. **Port validation**: Indicates missing validation in type converter before passing to ModulePortInfo

### Recommendation Justification

While Issue #8219 has the same assertion pattern, it's in a different dialect (ESI vs Moore) and context (bundle handling vs type conversion). This crash represents a **specific gap in string type handling** during Moore-to-Core conversion that warrants its own bug report.

**Status**: This should be reported as a **NEW ISSUE** with reference to Issue #8219 as a similar assertion pattern but different root cause.

---

## Next Steps

1. **Minimize test case** to smallest reproducible input
2. **Add type validation** in Moore TypeConverter before passing to getModulePortInfo
3. **Consider diagnostic** for unsupported string types in Moore imports
4. **Reference**: Issue #8219 demonstrates need for safer dyn_cast patterns

---

## Search Summary

- **Total Issues Scanned**: 50 open + ~100 closed in last 2 years
- **Keywords Matched**: 9 issues
- **Exact Assertion Match**: 1 issue (partially - different component)
- **Component Match**: 0 exact matches
- **Dialect Match**: 1 Moore-related but different assertion

---

*Report generated on 2026-01-31T17:36:20.307597*
