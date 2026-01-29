# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| **Recommendation** | `likely_new` |
| **Top Similarity Score** | 75/100 |
| **Most Similar Issue** | [#8283](https://github.com/llvm/circt/issues/8283) |

## Search Keywords

- `string port MooreToCore`
- `dyn_cast assertion MooreToCore`
- `ModulePortInfo crash`
- `sanitizeInOut`
- `Moore string`
- `DynamicStringType`
- `string port crash`
- `InOutType assertion`

## Similar Issues Found

### 1. [#8283](https://github.com/llvm/circt/issues/8283) - [ImportVerilog] Cannot compile forward declared string type
**Score: 75/100** | **State: OPEN**

**Match Reasons:**
- Same issue: string type not properly supported in Moore/MooreToCore
- Same error pattern: MooreToCore lacks string-type conversion
- Same root cause: string type variables fail during lowering
- Keywords match: string, Moore, MooreToCore

**Difference:** #8283 focuses on string *variables*, while this crash involves string-typed *module ports* — a potentially different code path through `ModulePortInfo` and `sanitizeInOut()`.

---

### 2. [#8332](https://github.com/llvm/circt/issues/8332) - [MooreToCore] Support for StringType from moore to llvm dialect
**Score: 70/100** | **State: OPEN**

**Match Reasons:**
- Related: discusses StringType support in MooreToCore
- Feature request for string type handling
- Keywords match: string, MooreToCore

**Difference:** This is a feature request/discussion, not a bug report for a specific crash.

---

### 3. [#8930](https://github.com/llvm/circt/issues/8930) - [MooreToCore] Crash with sqrt/floor
**Score: 45/100** | **State: OPEN**

**Match Reasons:**
- Same assertion message: `dyn_cast on a non-existent value`
- Same failing pass: MooreToCore (convert-moore-to-core)
- Same crash location: `Casting.h` dyn_cast

**Difference:** Triggered by real type conversion (sqrt/floor), not string port handling.

---

### 4. [#8173](https://github.com/llvm/circt/issues/8173) - [ImportVerilog] Crash on ordering-methods-reverse test
**Score: 40/100** | **State: OPEN**

**Match Reasons:**
- Related to string type handling
- Error: expression of type `!moore.string` cannot be cast

**Difference:** Different crash pattern, occurs during different phase.

---

### 5. [#4036](https://github.com/llvm/circt/issues/4036) - [PrepareForEmission] Crash when inout operations are passed to instance ports
**Score: 25/100** | **State: OPEN**

**Match Reasons:**
- InOutType assertion failure (different context)
- Port handling crash

**Difference:** Different dialect (HW vs Moore), different stage (PrepareForEmission vs MooreToCore).

---

## Recommendation

**`likely_new`** - This issue should likely be filed as a new bug report.

**Reasoning:**
1. While #8283 describes the same underlying limitation (string type not supported in MooreToCore), it focuses on string *variables*
2. This crash specifically involves **string-typed module ports**, which exercises a distinct code path through:
   - `getModulePortInfo()` 
   - `hw::ModulePortInfo` constructor
   - `sanitizeInOut()` with `dyn_cast<hw::InOutType>`
3. The specific crash location (`MooreToCore.cpp:259`, `sanitizeInOut()`) and assertion message are not documented in existing issues
4. Filing separately will help track this specific manifestation and ensure proper fix coverage

**Suggested Action:**
- File as new issue
- Reference #8283 as related issue
- Add labels: `Moore`, `bug`
