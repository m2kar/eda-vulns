# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 6 |
| Top Similarity Score | **11.5** |
| **Recommendation** | **review_existing** |

## Search Parameters

- **Dialect**: Moore
- **Failing Pass**: MooreToCorePass
- **Crash Type**: assertion
- **Keywords**: StringType, string, MooreToCore, convertType, port, assertion, dyn_cast, InOutType
- **Assertion**: `dyn_cast on a non-existent value`

## Top Similar Issues

### [#8930](https://github.com/llvm/circt/issues/8930) (Score: 11.5) ⚠️ HIGHEST MATCH

**Title**: [MooreToCore] Crash with sqrt/floor

**State**: OPEN

**Labels**: Moore

**Match Reasons**:
- ✅ **SAME assertion message**: `dyn_cast on a non-existent value`
- ✅ **SAME crash location**: `llvm/Support/Casting.h:650`
- ✅ Title contains MooreToCore
- ✅ Body mentions dyn_cast, assertion
- ✅ Has Moore label

**Analysis**: This issue has the **exact same assertion failure** but triggered by `real` type instead of `string` type. Both represent missing type conversion rules in MooreToCore.

---

### [#8332](https://github.com/llvm/circt/issues/8332) (Score: 9.0) ⚠️ DIRECT FEATURE REQUEST

**Title**: [MooreToCore] Support for StringType from moore to llvm dialect

**State**: OPEN

**Labels**: (none)

**Match Reasons**:
- ✅ Title explicitly mentions **StringType**
- ✅ Title contains MooreToCore
- ✅ Body discusses string type conversion strategy

**Analysis**: This is a **feature request** to add StringType support. Our crash is caused by the **same missing feature**. This issue proposes using `Int Type and RefType` to represent strings.

---

### [#8292](https://github.com/llvm/circt/issues/8292) (Score: 4.0)

**Title**: [MooreToCore] Support for Unsized Array Type

**State**: OPEN

**Match Reasons**:
- Title contains MooreToCore
- Body discusses type conversion challenges

---

### [#8476](https://github.com/llvm/circt/issues/8476) (Score: 3.5)

**Title**: [MooreToCore] Lower exponentiation to `math.ipowi`

**State**: OPEN

**Labels**: Moore, good first issue

**Match Reasons**:
- Title contains MooreToCore
- Has Moore label

---

### [#4036](https://github.com/llvm/circt/issues/4036) (Score: 1.0)

**Title**: [PrepareForEmission] Crash when inout operations are passed to instance ports

**State**: OPEN

**Match Reasons**:
- Body mentions InOutType

---

## Recommendation

**Action**: `review_existing`

⚠️ **Review Required**

Two highly related issues were found:

1. **#8930** - Has the **exact same assertion failure** (`dyn_cast on a non-existent value`) at the same crash location. However, it's triggered by `real` type, not `string` type.

2. **#8332** - Is a **feature request** for StringType support, which is exactly what's missing and causing our crash.

### Decision Matrix

| Scenario | Action |
|----------|--------|
| If #8332 covers StringType as port type | Comment on #8332 with this test case |
| If #8930 should track all "missing type" crashes | Comment on #8930 with this test case |
| If this is a distinct issue (StringType as port specifically) | Create new issue, reference #8332 and #8930 |

### Recommended Action

**Option A**: Add a comment to **#8332** since it's the direct feature request for StringType support. Include:
- The minimized test case (string as output port)
- The stack trace showing the crash path
- Note that this is another instance of missing type conversion

**Option B**: Create a new issue if the StringType-as-port scenario is considered distinct from the general StringType support discussion.

## Scoring Weights Used

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
| Same crash location | 2.0 | If crash site file:line matches |
