# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 31 |
| Top Similarity Score | 21 |
| **Recommendation** | **review_existing** |

## Search Parameters

- **Dialect**: Moore
- **Failing Pass**: MooreToCorePass
- **Crash Type**: assertion
- **Keywords**: string,StringType,DynamicStringType,ModulePortInfo,sanitizeInOut,MooreToCore,dyn_cast,port type,type conversion,InOutType

## Top Similar Issues

### [#9570](https://github.com/llvm/circt/issues/9570) - Score: 21

**Title**: # [Moore] Assertion in MooreToCore when module uses packed union type as port

**State**: OPEN

---

### [#8332](https://github.com/llvm/circt/issues/8332) - Score: 17

**Title**: [MooreToCore] Support for StringType from moore to llvm dialect

**State**: OPEN

---

### [#8283](https://github.com/llvm/circt/issues/8283) - Score: 14

**Title**: [ImportVerilog] Cannot compile forward decleared string type

**State**: OPEN

---

### [#8382](https://github.com/llvm/circt/issues/8382) - Score: 14

**Title**: [FIRRTL] Crash with fstring type on port

**State**: OPEN

---

### [#8930](https://github.com/llvm/circt/issues/8930) - Score: 12

**Title**: [MooreToCore] Crash with sqrt/floor

**State**: OPEN

---


## Recommendation

**Action**: `review_existing`


### ⚠️ Review Required

A highly similar issue was found. Please review the existing issue(s) before creating a new one.

**If the existing issue describes the same problem:**
- Add your test case as a comment
- Reference the original analysis in the comment
- Mark status as 'duplicate'

**If the issue is different:**
- Note the differences from existing issues
- Include reference to related issues in your report
- Proceed with bug report generation


## Scoring Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2 | Per keyword found in title |
| Body keyword match | 1 | Per keyword found in body |
| Assertion message match | 3 | If assertion matches |
| Dialect label match | 1 | If dialect label matches |
| Failing pass match | 2 | If failing pass appears in issue |

---
Generated: $(date)
