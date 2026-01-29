# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 10 |
| Top Similarity Score | 6.5 |
| **Recommendation** | **likely_new** |

## Search Parameters

- **Dialect**: Moore
- **Failing Pass**: MooreToCorePass
- **Crash Type**: assertion
- **Assertion Message**: `dyn_cast on a non-existent value`
- **Keywords**: string, DynamicStringType, isHWValueType, MooreToCore, SVModuleOpConversion, port type, unsupported type, InOutType

## Top Similar Issues

### [#8283](https://github.com/llvm/circt/issues/8283) (Score: 6.5)

**Title**: [ImportVerilog] Cannot compile forward decleared string type

**State**: OPEN

**Labels**: (none)

**Relevance**: **HIGH** - Same root cause area (string type not supported in MooreToCore)

**Key Difference**: That issue is about **string variables**, this bug is about **string PORTS**

**Match Reasons**:
- Title contains 'string' (weight: 2.0)
- Body contains 'string' (weight: 1.0)
- Body contains 'MooreToCore' (weight: 1.0)
- Body mentions string type conversion failure (weight: 1.5)
- Related to moore.variable with string type (weight: 1.0)

---

### [#8332](https://github.com/llvm/circt/issues/8332) (Score: 5.0)

**Title**: [MooreToCore] Support for StringType from moore to llvm dialect

**State**: OPEN

**Labels**: (none)

**Relevance**: **MEDIUM** - Discusses StringType lowering approach

**Key Difference**: Feature request for string support, not a crash report

**Match Reasons**:
- Title contains 'MooreToCore' (weight: 2.0)
- Title contains 'StringType' (weight: 2.0)
- Body discusses string type lowering (weight: 1.0)

---

### [#8930](https://github.com/llvm/circt/issues/8930) (Score: 4.5)

**Title**: [MooreToCore] Crash with sqrt/floor

**State**: OPEN

**Labels**: Moore

**Relevance**: **MEDIUM** - Same assertion message pattern

**Key Difference**: Different trigger (real type conversion via `moore.ConversionOp`, not string port)

**Match Reasons**:
- Title contains 'MooreToCore' (weight: 2.0)
- Same assertion message 'dyn_cast on a non-existent value' (weight: 3.0 - partial)
- Moore dialect label (weight: 1.5)

**Note**: Same assertion but different root cause - that crash happens in `ConversionOpConversion`, this one happens in `getModulePortInfo`

---

### [#8825](https://github.com/llvm/circt/issues/8825) (Score: 3.0)

**Title**: [LLHD] Switch from hw.inout to a custom signal reference type

**State**: OPEN

**Labels**: LLHD

**Relevance**: **LOW** - Related to type system but different dialect

**Match Reasons**:
- Body mentions 'isHWValueType' (weight: 2.0)
- Body mentions 'hw.inout' type constraints (weight: 1.0)

---

### [#8173](https://github.com/llvm/circt/issues/8173) (Score: 2.5)

**Title**: [ImportVerilog] Crash on ordering-methods-reverse test

**State**: OPEN

**Labels**: bug, ImportVerilog

**Relevance**: **LOW** - String array handling issue

**Match Reasons**:
- Body contains 'string' array handling (weight: 1.0)
- Different crash location (bit vector cast error)

---

## Recommendation

**Action**: `likely_new`

📋 **Proceed with Caution**

Related issues exist but this appears to be a **different manifestation** of the string type support gap:

| Issue | Focus | Crash Location |
|-------|-------|----------------|
| #8283 | String **variable** | `moore.variable` legalization |
| #8332 | String type **lowering design** | N/A (feature request) |
| #8930 | Real type **conversion** | `ConversionOpConversion` |
| **This bug** | String **PORT** | `getModulePortInfo` → `HWModuleOp::build` |

### Why This Is Likely New

1. **Different crash path**: Crashes in `getModulePortInfo()` when building HW module ports, not in variable handling
2. **Specific trigger**: Using `string` type as **module output port** (not as internal variable)
3. **Missing validation**: No existing issue mentions the port type validation gap where `isHWValueType(DynamicStringType)` returns false

### Recommended Actions

1. **Proceed** to generate the bug report
2. **Reference** related issues (#8283, #8332) in the report
3. **Highlight** that this is specifically about port type validation, not general string support

## Scoring Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
| Failing pass match | 2.0 | If failing pass appears in issue |
