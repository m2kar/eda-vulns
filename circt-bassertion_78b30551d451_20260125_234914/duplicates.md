# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 18 |
| Top Similarity Score | 7.0 |
| **Recommendation** | **likely_new** |

## Search Parameters

- **Dialect**: Moore
- **Failing Pass**: MooreToCore
- **Crash Type**: assertion
- **Assertion Message**: `dyn_cast on a non-existent value`
- **Keywords**: string, StringType, DynamicStringType, port, getModulePortInfo, sanitizeInOut, InOutType, dyn_cast, MooreToCore, SVModuleOpConversion

## Top Similar Issues

### [#8332](https://github.com/llvm/circt/issues/8332) (Score: 7.0)

**Title**: [MooreToCore] Support for StringType from moore to llvm dialect

**State**: OPEN

**Labels**: (none)

**Match Reasons**:
- Title: MooreToCore ✓
- Title: StringType ✓
- Body: string, StringType ✓

**Relevance**: This is a **feature request** discussing how to implement StringType support in MooreToCore conversion. It is NOT a bug report about the specific port-related crash we found.

---

### [#8930](https://github.com/llvm/circt/issues/8930) (Score: 6.5)

**Title**: [MooreToCore] Crash with sqrt/floor

**State**: OPEN

**Labels**: Moore

**Match Reasons**:
- Title: MooreToCore ✓
- Body: exact assertion message "dyn_cast on a non-existent value" ✓
- Label: Moore ✓

**Relevance**: Same assertion message, but **different crash context**:
- #8930: crashes in `ConversionOpConversion` (real type conversion)
- Our bug: crashes in `SVModuleOpConversion/getModulePortInfo` (string type as module port)

---

### [#8176](https://github.com/llvm/circt/issues/8176) (Score: 3.5)

**Title**: [MooreToCore] Crash when getting values to observe

**State**: OPEN

**Labels**: Moore

---

### [#8269](https://github.com/llvm/circt/issues/8269) (Score: 3.5)

**Title**: [MooreToCore] Support `real` constants

**State**: OPEN

**Labels**: Moore

---

### [#7535](https://github.com/llvm/circt/issues/7535) (Score: 2.0)

**Title**: [MooreToCore] VariableOp lowered failed

**State**: OPEN

**Labels**: (none)

---

## Recommendation

**Action**: `likely_new`

📋 **Proceed with Caution**

Related issues exist but this appears to be a different bug.

### Key Differentiators

| Aspect | Our Bug | #8332 | #8930 |
|--------|---------|-------|-------|
| Type | Crash/Assertion | Feature Request | Crash/Assertion |
| Trigger | String as **module port** | General StringType support | sqrt/floor with real type |
| Crash Location | `getModulePortInfo` → `sanitizeInOut` | N/A | `ConversionOpConversion` |
| Type Involved | `sim::DynamicStringType` | `moore::StringType` | `moore::real` |

### Recommended Actions

1. **Proceed to generate the bug report**
2. **Reference related issues** in the report:
   - Mention #8332 as related StringType work
   - Note that #8930 has similar assertion but different cause
3. **Highlight what makes this bug different**:
   - Specific to string type used as module PORT
   - Crashes in `sanitizeInOut` during port info gathering
   - Root cause: `hw::ModulePortInfo` cannot handle `sim::DynamicStringType`

## Scoring Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
| Failing pass match | 2.0 | If failing pass appears in issue |
