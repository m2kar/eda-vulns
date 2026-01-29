# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 10 |
| Top Similarity Score | 10.5 |
| **Recommendation** | **review_existing** |

## Search Parameters

- **Dialect**: Moore
- **Failing Pass**: MooreToCore
- **Crash Type**: assertion
- **Keywords**: string, DynamicStringType, ModulePortInfo, InOutType, dyn_cast, MooreToCore, port, sanitizeInOut, type conversion, hw::PortInfo
- **Assertion Message**: `dyn_cast on a non-existent value`

## Top Similar Issues

### [#8930](https://github.com/llvm/circt/issues/8930) (Score: 10.5)

**Title**: [MooreToCore] Crash with sqrt/floor

**State**: OPEN

**Labels**: Moore

**Match Details**: 
- Same assertion message: `dyn_cast on a non-existent value`
- Same failing pass: MooreToCore
- Same crash pattern: type conversion issue with non-hardware types

---

### [#8269](https://github.com/llvm/circt/issues/8269) (Score: 7.5)

**Title**: [MooreToCore] Support real constants

**State**: OPEN

**Labels**: Moore

---

### [#9206](https://github.com/llvm/circt/issues/9206) (Score: 6.5)

**Title**: [ImportVerilog] moore.conversion generated instead of moore.int_to_string

**State**: OPEN

**Labels**: Moore, ImportVerilog

---

### [#8973](https://github.com/llvm/circt/issues/8973) (Score: 5.5)

**Title**: [MooreToCore] Lowering to math.ipow?

**State**: OPEN

**Labels**: Moore

---

### [#8476](https://github.com/llvm/circt/issues/8476) (Score: 5.5)

**Title**: [MooreToCore] Lower exponentiation to math.ipowi

**State**: OPEN

**Labels**: Moore, good first issue

---

## Recommendation

**Action**: `review_existing`

⚠️ **Review Required**

A highly similar issue was found (Issue #8930, score 10.5). Please review the existing issue before creating a new one.

**Key Similarity with #8930:**
- **Same assertion message**: Both crash with `dyn_cast on a non-existent value`
- **Same failing pass**: MooreToCore conversion
- **Same root cause category**: Non-hardware types (real vs string) failing during type conversion

**However, this may still be a distinct bug because:**
- #8930: Crash with `moore.real` type (sqrt/floor operations)
- This bug: Crash with `string` type on module port (DynamicStringType via sanitizeInOut)
- Different crash location: #8930 in ConversionOp, this in getModulePortInfo/sanitizeInOut

**Recommended Action:**
1. Review Issue #8930 to understand if the fix would also address this crash
2. If the root cause is different (port type validation vs operation type conversion):
   - Proceed to generate a new bug report
   - Reference #8930 as a related issue with similar symptoms
3. If the root cause is the same:
   - Add this test case as a comment to #8930

## Scoring Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
| Failing pass match | 2.0 | If failing pass appears in issue |
