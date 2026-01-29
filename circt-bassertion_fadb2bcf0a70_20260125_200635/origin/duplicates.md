# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| **Recommendation** | `likely_new` |
| **Top Similarity Score** | 8.5 / 15.0 |
| **Most Similar Issue** | #8930 |
| **Total Issues Searched** | 25 |

## Search Keywords

- `string`, `DynamicStringType`, `ModulePortInfo`, `sanitizeInOut`, `InOutType`, `MooreToCore`, `port type`, `dyn_cast`
- Assertion: `"dyn_cast on a non-existent value"`

## Scoring Methodology

| Factor | Weight |
|--------|--------|
| Title keyword match | 2.0 |
| Body keyword match | 1.0 |
| Assertion message match | 3.0 |
| Dialect label match | 1.5 |

## Potential Duplicates

### 1. [#8930 - [MooreToCore] Crash with sqrt/floor](https://github.com/llvm/circt/issues/8930) ⚠️ HIGH

**Score: 8.5** | State: `open` | Labels: `Moore`

**Similarity Analysis:**
- ✅ Same assertion: `"dyn_cast on a non-existent value"`
- ✅ Same pass: MooreToCore
- ❌ Different trigger: sqrt/floor operations vs string port type
- ❌ Different root cause: ConversionOp vs ModulePortInfo

**Verdict:** Related but NOT duplicate - same assertion pattern but different code path and trigger.

---

### 2. [#8332 - [MooreToCore] Support for StringType from moore to llvm dialect](https://github.com/llvm/circt/issues/8332)

**Score: 6.0** | State: `open` | Labels: none

**Similarity Analysis:**
- ✅ Discusses StringType in MooreToCore
- ❌ Focus on lowering to LLVM dialect
- ❌ No crash/assertion mentioned
- ❌ Different scope (feature request vs bug)

**Verdict:** Related feature discussion, NOT duplicate.

---

### 3. [#8283 - [ImportVerilog] Cannot compile forward declared string type](https://github.com/llvm/circt/issues/8283)

**Score: 5.0** | State: `open` | Labels: none

**Similarity Analysis:**
- ✅ String type handling issue
- ✅ Mentions MooreToCore lacks string-type conversion
- ❌ Different error: "failed to legalize operation" vs assertion
- ❌ Different trigger: variable declaration vs port

**Verdict:** Related string handling issue, NOT duplicate.

---

### 4. [#7535 - [MooreToCore] VariableOp lowered failed](https://github.com/llvm/circt/issues/7535)

**Score: 4.5** | State: `open` | Labels: none

**Similarity Analysis:**
- ✅ MooreToCore crash
- ✅ Mentions hw::InOutType casting
- ❌ Different trigger: struct type vs string port
- ❌ Different stack trace

**Verdict:** Different bug, NOT duplicate.

---

### 5. [#8219 - [ESI] Assertion: dyn_cast on a non-existent value](https://github.com/llvm/circt/issues/8219)

**Score: 4.0** | State: `closed` | Labels: `ESI`

**Similarity Analysis:**
- ✅ Same assertion message
- ❌ Different dialect (ESI vs Moore)
- ❌ Already closed/fixed

**Verdict:** Same assertion pattern in different context, NOT duplicate.

---

## Conclusion

**Recommendation: `likely_new`**

This bug appears to be a **new issue**. While issue #8930 shares the same assertion message, the root cause is different:

| Aspect | This Bug | #8930 |
|--------|----------|-------|
| Trigger | `output string` port | `moore.conversion` with real type |
| Failing function | `getModulePortInfo()` | `ConversionOpConversion` |
| Type involved | `sim::DynamicStringType` | `moore.real` |
| Code path | Port type handling | Type conversion |

The bug is specifically about **string type used as module port**, which causes `ModulePortInfo::sanitizeInOut()` to fail when it encounters `sim::DynamicStringType` (not a valid HW port type).

## Related Issues for Reference

- #8332: StringType support discussion (feature)
- #8283: String variable compilation issue
- #8930: Similar assertion, different cause
