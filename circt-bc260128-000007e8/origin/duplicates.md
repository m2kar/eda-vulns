# CIRCT Duplicate Issue Analysis Report
**Crash ID:** 260128-000007e8  
**Date:** 2026-01-31  
**Analysis Type:** Duplicate Search & Similarity Assessment

---

## Executive Summary

The reported crash is a **high-confidence duplicate or direct consequence** of existing issue **#8332**. The root cause is insufficient StringType support in the MooreToCore conversion pass, which prevents string-typed module ports from being properly converted, leaving port types invalid/null. When `sanitizeInOut()` attempts to cast these invalid types to `InOutType`, it triggers a `dyn_cast` assertion.

**Recommendation:** Review existing issue #8332 "[MooreToCore] Support for StringType from moore to llvm dialect" as the primary tracking issue. Consider reporting this crash as a concrete test case demonstrating the impact.

---

## Search Strategy

### Keywords Extracted
- Primary focus: `string`, `port`, `InOutType`, `dyn_cast`, `sanitizeInOut`
- Secondary scope: `MooreToCore`, `SVModuleOp`, `Moore` dialect, `assertion`

### Search Methods
1. Direct keyword searches via `gh issue list --search`
2. Filtered by labels (Moore dialect, MooreToCore conversion)
3. Focused on open issues to identify ongoing work

### Coverage
- Repository: llvm/circt
- Scope: All open issues in Moore and MooreToCore areas
- Result: 5 highly relevant issues identified

---

## Detailed Candidate Analysis

### 🔴 **#8332: [MooreToCore] Support for StringType from moore to llvm dialect**

**Similarity Score:** 8.5/10  
**State:** OPEN  
**URL:** https://github.com/llvm/circt/issues/8332

#### Match Rationale
- **Keywords:** StringType, MooreToCore, conversion, moore.variable, type legalization
- **Root Cause Match:** Identical - lack of StringType conversion support in MooreToCore
- **Impact:** Direct consequence of this feature gap

#### Description Summary
Issue tracking request to add StringType support in the Moore-to-LLVM conversion. Current implementation lacks conversion rules for StringType, leaving variables with string types unconvertible. This is precisely the mechanism that causes our crash: when a module port is declared as `output string`, the MooreToCore pass cannot convert it, leaving the port type empty/invalid.

#### Connection to Our Crash
```
Our crash flow:
  module output string port
    ↓
  MooreToCore conversion fails (no StringType support) 
    ↓
  Port type left as invalid/empty value
    ↓
  sanitizeInOut() attempts: llvm::dyn_cast<InOutType>(invalid_type)
    ↓
  ASSERTION FAILURE: "dyn_cast on a non-existent value"
```

---

### 🟠 **#8283: [ImportVerilog] Cannot compile forward decleared string type**

**Similarity Score:** 8.0/10  
**State:** OPEN  
**URL:** https://github.com/llvm/circt/issues/8283

#### Match Rationale
- **Keywords:** string type, MooreToCore, moore.variable, legalization failure
- **Error Pattern:** Same root cause, different manifestation point
- **Test Case:** String variable in SystemVerilog

#### Description Summary
Reports inability to legalize `moore.variable` operations with string type. Compilation fails with:
```
error: failed to legalize operation 'moore.variable'
%0 = "moore.variable"() <{name = "str"}> : () -> !moore.ref<string>
```

The note explicitly states: "MooreToCore's lack of string-type conversion" - identical diagnosis to our issue.

#### Distinction from Our Issue
- This issue manifests at the **variable level** (moore.variable operation)
- Our crash manifests at the **module port level** (SVModuleOp conversion)
- Both stem from the **same root cause**: No StringType → HW Type conversion

---

### 🟡 **#8176: [MooreToCore] Crash when getting values to observe**

**Similarity Score:** 6.5/10  
**State:** OPEN  
**URL:** https://github.com/llvm/circt/issues/8176

#### Match Rationale
- **Keywords:** MooreToCore crash, dyn_cast assertion pattern
- **Error Mechanism:** Same assertion type (dyn_cast on invalid value)
- **Context:** MooreToCore conversion pipeline

#### Description Summary
Another MooreToCore crash during type handling, but triggered in a different context (getValuesToObserve function with always_comb procedure). The common pattern is **dyn_cast assertions on invalid type values** during MooreToCore conversion.

#### Significance
Suggests a broader pattern of type safety issues in MooreToCore when encountering unsupported or partially-converted type scenarios.

---

### 🟡 **#8930: [MooreToCore] Crash with sqrt/floor**

**Similarity Score:** 6.0/10  
**State:** OPEN  
**URL:** https://github.com/llvm/circt/issues/8930

#### Match Rationale
- **Keywords:** MooreToCore crash, dyn_cast assertion, ConversionOp
- **Error Mechanism:** dyn_cast on non-existent value
- **Context:** Type conversion during operation lowering

#### Description Summary
MooreToCore crashes when converting `moore.conversion` operation involving `real` types. Stack trace shows assertion in `getBitWidth()` attempting dyn_cast on an IntegerType.

#### Connection
Demonstrates another instance of MooreToCore's fragility when handling type conversions for non-standard types (real, string).

---

### 🟢 **#7531: [Moore] Input triggers assertion in canonicalizer infra**

**Similarity Score:** 4.5/10  
**State:** OPEN  
**URL:** https://github.com/llvm/circt/issues/7531

#### Match Rationale
- **Keywords:** Moore, assertion, canonicalizer
- **Error Type:** Assertion failure (different mechanism)

#### Description Summary
Canonicalizer assertion triggered by specific Moore IR pattern with array types. Different from our crash (canonicalizer vs. port sanitization).

#### Distinction
- Lower relevance due to different crash location and mechanism
- Included for completeness in Moore dialect issue tracking

---

## Similarity Scoring Methodology

| Factor | Weight | Notes |
|--------|--------|-------|
| Keywords Match | 2 pts | string, MooreToCore, StringType presence |
| Crash Type Match | 2 pts | Assertion/dyn_cast on invalid value |
| Root Cause Match | 3 pts | Type conversion failure in MooreToCore |
| Stack Trace Similarity | 2 pts | Similar location in conversion pipeline |
| **Total** | **10 pts** | Max possible score |

### Scoring Results
- **#8332:** Keywords(2) + Crash(2) + RootCause(3) + Trace(1.5) = **8.5**
- **#8283:** Keywords(2) + Crash(2) + RootCause(2.5) + Trace(1.5) = **8.0**
- **#8176:** Keywords(1.5) + Crash(2) + RootCause(1.5) + Trace(1.5) = **6.5**
- **#8930:** Keywords(1.5) + Crash(2) + RootCause(1.5) + Trace(1.5) = **6.0**
- **#7531:** Keywords(1) + Crash(1.5) + RootCause(0) + Trace(2) = **4.5**

---

## Root Cause Alignment

### Our Crash
```
Assertion at: circt::hw::ModulePortInfo::sanitizeInOut() 
             llvm::dyn_cast<InOutType>(port_type)
             
Trigger: port_type is invalid/empty (null value)
         because MooreToCore cannot convert "string" type
```

### Issue #8332 (Best Match)
```
Problem: MooreToCore lacks StringType conversion rules
         StringConstantOp and FormatOps are insufficient
Impact: StringType variables/ports fail to convert
        leaving them as unconvertible "!moore.ref<string>" types
```

### The Connection
- **#8332 describes the missing feature** (StringType support)
- **Our crash demonstrates the consequence** when that feature is missing at the module port level
- **#8283 demonstrates the same consequence** at the variable level

---

## Recommendation: `review_existing`

### Rationale
1. **High-confidence mapping to existing issue #8332**
   - Root cause is well-documented
   - Ongoing feature development likely expected
   
2. **Not a completely new bug**
   - The crash is expected behavior when StringType is unsupported
   - Issue #8332 already tracks the feature request
   
3. **Value of reporting this case**
   - Provides concrete test case for module port-level impact
   - Could serve as additional motivation for feature development
   - Demonstrates the feature gap's severity across multiple contexts

### Recommended Action
- **Do not open new issue** as primary report
- **Cross-reference this crash** in issue #8332 as a concrete example
- **Add as test case** once StringType support is implemented
- **Consider as regression test** to ensure string port handling doesn't break

---

## Supporting Evidence

### Test Case Characteristics
- **Input:** SystemVerilog module with `output string` port
- **Operation:** Declared module port of string type with value assignments
- **Expected:** Proper conversion or diagnostic error
- **Actual:** Assertion failure in sanitizeInOut due to invalid type

### Error Signature
```
Failed assertion: detail::isPresent(Val) && "dyn_cast on a non-existent value"
Location: llvm::dyn_cast<InOutType> in PortImplementation.h:177
```

This is a **hard assertion**, indicating:
- Not a graceful error handling path
- Type value is genuinely null/missing (not just wrong type)
- Indicates upstream conversion produced invalid state

---

## Conclusion

The reported crash **260128-000007e8** is a direct consequence of missing StringType support in the MooreToCore conversion pass, as tracked in issue **#8332**. 

**Classification:** Likely Duplicate / Related to #8332  
**Action:** Review existing issue and cross-reference; do not create new primary issue  
**Priority:** Should be resolved once #8332 StringType support is implemented

The crash should be used as additional evidence for the importance of completing StringType support implementation in MooreToCore, particularly for module port-level constructs.

---

**Report Generated:** 2026-01-31  
**Analyzed by:** check-duplicates worker  
**Status:** Ready for developer review
