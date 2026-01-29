# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 3 |
| Top Similarity Score | 6.0 |
| **Recommendation** | **likely_new** |

## Search Parameters

- **Dialect**: Moore
- **Failing Pass**: MooreToCore
- **Crash Type**: assertion
- **Keywords**: string, StringType, MooreToCore, port, type conversion

## Top Similar Issues

### [#8332](https://github.com/llvm/circt/issues/8332) (Score: 6.0)

**Title**: [MooreToCore] Support for StringType from moore to llvm dialect

**State**: OPEN

**Type**: Feature Request (not crash report)

**Summary**: This issue discusses adding StringType support for simulation purposes. It proposes using IntType and RefType to represent string data internally, with dynamic memory allocation for variable-length strings.

**Relation to our bug**: This confirms that StringType is not currently supported, but it's a feature request, not a crash report. Our bug demonstrates that the lack of support causes assertion failure instead of graceful error handling.

---

### [#8382](https://github.com/llvm/circt/issues/8382) (Score: 3.0)

**Title**: [FIRRTL] Crash with fstring type on port

**State**: OPEN

**Type**: Crash Report

**Summary**: Similar crash when using fstring type on FIRRTL module port. Suggests IR verifier should catch unsupported types.

**Relation to our bug**: Similar pattern (unsupported type on port causes crash), but different dialect (FIRRTL vs Moore).

---

### [#8292](https://github.com/llvm/circt/issues/8292) (Score: 2.0)

**Title**: [MooreToCore] Support for Unsized Array Type

**State**: OPEN

**Type**: Feature Request

**Summary**: Request for unsized array type support in MooreToCore.

**Relation to our bug**: Same pass (MooreToCore), similar pattern (missing type conversion).

---

## Recommendation

**Action**: `likely_new`

📋 **Proceed with Caution**

Related issues exist but this appears to be a distinct bug:

1. **#8332 is a feature request**, not a crash report. It discusses how to implement StringType support.

2. **Our bug is a crash report** demonstrating that:
   - Using `string` type ports causes assertion failure
   - CIRCT crashes instead of gracefully handling unsupported types
   - This is a usability/robustness bug, not just a missing feature

3. **Different scope**:
   - #8332: How to lower StringType to LLVM for simulation
   - Our bug: MooreToCore crashes when StringType appears in port list

**Recommended Actions**:
- Proceed to generate the bug report
- Reference #8332 as related context (StringType support discussion)
- Reference #8382 as similar pattern (type on port crash)
- Emphasize that the issue is ungraceful crash, not just missing feature

## Scoring Weights Used

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
| Failing pass match | 2.0 | If failing pass appears in issue |
