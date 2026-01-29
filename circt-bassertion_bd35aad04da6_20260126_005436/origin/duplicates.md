# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 5 |
| Top Similarity Score | **8.5** |
| **Recommendation** | **review_existing** |

## Search Parameters

- **Dialect**: Moore
- **Failing Pass**: MooreToCore
- **Crash Type**: assertion
- **Keywords**: StringType, MooreToCore, string port, dyn_cast, TypeConverter, getModulePortInfo, non-existent value
- **Assertion Message**: `dyn_cast on a non-existent value`

## Top Similar Issues

### [#8332](https://github.com/llvm/circt/issues/8332) (Score: 8.5) ⚠️ EXACT MATCH

**Title**: [MooreToCore] Support for StringType from moore to llvm dialect

**State**: OPEN

**Labels**: (none)

**Match Reasons**:
- Title contains: StringType (2.0), MooreToCore (2.0)
- Body discusses: StringType, moore, llvm, conversionOp (2.0)
- Exact feature match: StringType support in MooreToCore (2.5)

**Analysis**: This issue directly discusses adding StringType support from Moore to LLVM dialect. Our crash occurs because `StringType` has no type converter in `MooreToCore`, causing `convertType()` to return null, which then triggers `dyn_cast on a non-existent value` assertion.

---

### [#8930](https://github.com/llvm/circt/issues/8930) (Score: 6.5)

**Title**: [MooreToCore] Crash with sqrt/floor

**State**: OPEN

**Labels**: Moore

**Match Reasons**:
- Title contains: MooreToCore (2.0)
- Body contains: same assertion message `dyn_cast on a non-existent value` (3.0)
- Label: Moore (1.5)

**Analysis**: Different root cause (sqrt/floor operations) but same assertion symptom. Not a duplicate, but shows a pattern of missing type converters in MooreToCore.

---

### [#8269](https://github.com/llvm/circt/issues/8269) (Score: 3.5)

**Title**: [MooreToCore] Support `real` constants

**State**: OPEN

**Labels**: Moore

**Match Reasons**:
- Title contains: MooreToCore (2.0)
- Label: Moore (1.5)

---

### [#8476](https://github.com/llvm/circt/issues/8476) (Score: 3.5)

**Title**: [MooreToCore] Lower exponentiation to `math.ipowi`

**State**: OPEN

**Labels**: good first issue, Moore

---

### [#8292](https://github.com/llvm/circt/issues/8292) (Score: 2.0)

**Title**: [MooreToCore] Support for Unsized Array Type

**State**: OPEN

---

## Recommendation

**Action**: `review_existing`

⚠️ **HIGH SIMILARITY - Review Required**

Issue **#8332** directly addresses the missing StringType support in MooreToCore conversion, which is the exact cause of this crash.

### What to do:

**If Issue #8332 covers this crash:**
- Add this test case as a comment to #8332
- Mark this bug as `duplicate` in status.json
- Reference: "This crash is another manifestation of missing StringType converter"

**If the issues differ:**
- Proceed to generate bug report
- Reference #8332 as related issue
- Clarify: "While #8332 discusses general StringType lowering, this crash specifically occurs in `getModulePortInfo()` when StringType is used as a module port"

## Scoring Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
| Failing pass match | 2.0 | If failing pass appears in issue |
| Exact feature match | 2.5 | If discussing the same feature/limitation |
