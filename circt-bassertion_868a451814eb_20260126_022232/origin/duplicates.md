# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 15 |
| Top Similarity Score | 12.5 |
| **Recommendation** | **review_existing** |

## Search Parameters

- **Dialect**: Moore
- **Failing Pass**: MooreToCore
- **Crash Type**: assertion
- **Keywords**: string, output, port, MooreToCore, InOutType, dyn_cast, convertType, DynamicStringType, sanitizeInOut, SVModuleOp
- **Assertion**: `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed`

## Top Similar Issues

### [#8930](https://github.com/llvm/circt/issues/8930) (Score: 12.5)

**Title**: [MooreToCore] Crash with sqrt/floor

**State**: OPEN

**Labels**: Moore

**Match Reasons**:
- Same assertion message: `dyn_cast on a non-existent value`
- Same failing pass: MooreToCore
- Same dialect: Moore
- Both involve type conversion failures

---

### [#8276](https://github.com/llvm/circt/issues/8276) (Score: 9.0)

**Title**: [MooreToCore] Support for UnpackedArrayType emission

**State**: OPEN

**Labels**: (none)

**Match Reasons**:
- Same failing pass: MooreToCore
- Involves type conversion issues

---

### [#8973](https://github.com/llvm/circt/issues/8973) (Score: 8.5)

**Title**: [MooreToCore] Lowering to math.ipow?

**State**: OPEN

**Labels**: Moore

**Match Reasons**:
- Same failing pass: MooreToCore
- Same dialect: Moore

---

### [#8476](https://github.com/llvm/circt/issues/8476) (Score: 8.5)

**Title**: [MooreToCore] Lower exponentiation to `math.ipowi`

**State**: OPEN

**Labels**: Moore

**Match Reasons**:
- Same failing pass: MooreToCore
- Same dialect: Moore

---

### [#8163](https://github.com/llvm/circt/issues/8163) (Score: 8.5)

**Title**: [MooreToCore] Out-of-bounds moore.extract lowered incorrectly

**State**: OPEN

**Labels**: Moore

**Match Reasons**:
- Same failing pass: MooreToCore
- Same dialect: Moore

---

## Recommendation

**Action**: `review_existing`

⚠️ **Review Required**

A highly similar issue was found (Score: 12.5). Issue #8930 has the **exact same assertion message**:
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed
```

**Key Differences**:
- #8930: Crashes with `sqrt/floor` and `real` type conversion
- This bug: Crashes with `string` type output port

Both appear to be **related but different manifestations** of the same underlying problem: MooreToCore type conversion returning null/invalid types that cause dyn_cast failures.

**Recommended Actions**:
1. Review Issue #8930 to understand if the root cause is the same
2. If same root cause: Add this test case as a comment to #8930
3. If different root cause: Create new issue, referencing #8930 as related

## Scoring Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
| Failing pass match | 2.0 | If failing pass appears in issue |
