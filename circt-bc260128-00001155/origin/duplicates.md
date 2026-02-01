# Duplicate Check Report

**Generated**: 2026-02-01 05:18:51

## Summary

| Metric | Value |
|--------|-------|
| **Issues Found** | 70 |
| **Top Similarity Score** | 12.0 |
| **Recommendation** | **REVIEW_EXISTING** |
| **Confidence** | **HIGH** |

## Search Parameters

- **Dialect**: `Moore`
- **Failing Pass**: `MooreToCore`
- **Crash Type**: `assertion`
- **Keywords**: `StringType`, `MooreToCore`, `sanitizeInOut`, `dyn_cast non-existent`, `type conversion`, `module port`
- **Assertion Message**: `dyn_cast on a non-existent value`

## Top Similar Issues

### 1. [#9570](https://github.com/llvm/circt/issues/9570) - Score: 12.0

**Title**: [Moore] Assertion in MooreToCore when module uses packed union type as port

**State**: `OPEN`

---

### 2. [#8930](https://github.com/llvm/circt/issues/8930) - Score: 10.5

**Title**: [MooreToCore] Crash with sqrt/floor

**State**: `OPEN`

---

### 3. [#9572](https://github.com/llvm/circt/issues/9572) - Score: 10.0

**Title**: [Moore] Assertion failure when module has string type output port

**State**: `OPEN`

---

### 4. [#8332](https://github.com/llvm/circt/issues/8332) - Score: 7.0

**Title**: [MooreToCore] Support for StringType from moore to llvm dialect

**State**: `OPEN`

---

### 5. [#7627](https://github.com/llvm/circt/issues/7627) - Score: 6.5

**Title**: [MooreToCore] Unpacked array causes crash

**State**: `CLOSED`

---

## Recommendation

⚠️ **REVIEW REQUIRED - HIGH SIMILARITY FOUND**

A highly similar issue exists. Please review the existing issue(s) before creating a new one.

**Action Items:**
1. Compare this crash with the top similar issue(s)
2. If the issue is identical:
   - Mark as duplicate in status.json
   - Add your test case as a comment to the existing issue
3. If the issue is different:
   - Proceed to generate the bug report
   - Reference the related issue in your report

## Scoring Breakdown

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion matches body |
| Dialect label match | 1.5 | If dialect label matches |
| Failing pass match | 2.0 | If pass name appears |