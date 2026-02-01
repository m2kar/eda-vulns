# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 4 |
| Top Similarity Score | 7.0 |
| **Recommendation** | **likely_new** |

## Search Parameters

- **Dialect**: moore
- **Failing Pass**: SVModuleOpConversion
- **Crash Type**: assertion
- **Keywords**: union, packed union, module port, MooreToCore, sanitizeInOut, dyn_cast, assertion, ModulePortInfo, SVModuleOpConversion
- **Assertion Message**: `dyn_cast on a non-existent value in circt::hw::ModulePortInfo::sanitizeInOut()`

## Top Similar Issues

### [#8930](https://github.com/llvm/circt/issues/8930) (Score: 7.0)

**Title**: [MooreToCore] Crash with sqrt/floor

**State**: OPEN

**Labels**: Moore

**Similarity Breakdown**:
- Title keyword match: 2.0
- Body keyword match: 1.0
- Assertion message match: 3.0
- Dialect label match: 1.5
- Failing pass match: 0.0

**Notes**: Same assertion message but different trigger (sqrt/floor vs union port)

---

### [#8471](https://github.com/llvm/circt/issues/8471) (Score: 2.0)

**Title**: [ImportVerilog] Union type in call

**State**: OPEN

**Labels**: (none)

**Similarity Breakdown**:
- Title keyword match: 2.0
- Body keyword match: 0.0
- Assertion message match: 0.0
- Dialect label match: 0.0
- Failing pass match: 0.0

**Notes**: Related to union type but different issue (type mismatch in call)

---

### [#7378](https://github.com/llvm/circt/issues/7378) (Score: 2.0)

**Title**: [HW] Roundtrip test fail for !hw.union

**State**: OPEN

**Labels**: bug, HW

**Similarity Breakdown**:
- Title keyword match: 2.0
- Body keyword match: 0.0
- Assertion message match: 0.0
- Dialect label match: 0.0
- Failing pass match: 0.0

**Notes**: Related to hw.union but different issue (roundtrip test failure)

---

### [#8973](https://github.com/llvm/circt/issues/8973) (Score: 1.5)

**Title**: [MooreToCore] Lowering to math.ipow?

**State**: OPEN

**Labels**: Moore

**Similarity Breakdown**:
- Title keyword match: 0.0
- Body keyword match: 0.0
- Assertion message match: 0.0
- Dialect label label match: 1.5
- Failing pass match: 0.0

**Notes**: MooreToCore related but different issue (power operation lowering)

---

## Recommendation

**Action**: `likely_new`

📋 **Proceed with Caution**

Related issues exist but this appears to be a different bug.

**Recommended:**
- Proceed to generate bug report
- Reference related issues in report
- Highlight what makes this bug different

**Key Differences**:
1. **Trigger**: Packed union as module port (vs sqrt/floor operations)
2. **Dialect**: Moore dialect union type (vs real type conversion)
3. **Location**: `ModulePortInfo::sanitizeInOut()` (vs `ConversionOpConversion`)

## Scoring Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
| Failing pass match | 2.0 | If failing pass appears in issue |

## Similarity Score Interpretation

| Score Range | Meaning | Recommended Action |
|------------|----------|-------------------|
| >= 8.0 | High similarity | Review existing issue before creating new one |
| 4.0 - 7.9 | Related | Proceed but reference related issues |
| < 4.0 | Low similarity | Likely new issue |

## Related Issues to Reference

When creating the bug report, consider referencing:
- **#8930** - Same assertion failure pattern in MooreToCore conversion
- **#8471** - Union type support issues
- **#7378** - HW dialect union type handling
