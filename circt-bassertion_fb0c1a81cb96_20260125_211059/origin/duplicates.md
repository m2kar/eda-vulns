# Duplicate Check Report

## Summary
Searched CIRCT GitHub issues for similar problems. Found one related issue about StringType support (#8332), but it addresses a different aspect (type support vs. crash). No exact duplicates found.

## Search Queries

1. `"string port"` - No matches
2. `"sanitizeInOut"` - No matches
3. `"InOutType string"` - No matches
4. `"MooreToCore string"` - Found 2 issues:
   - #8283: "[ImportVerilog] Cannot compile forward declared string type" (2025-03-04)
   - #8292: "[MooreToCore] Support for Unsized Array Type" (2025-03-11)
   - **#8332: "[MooreToCore] Support for StringType from moore to llvm dialect" (2025-03-20)**
5. `"ModulePortInfo crash"` - No matches
6. `"non-existent value"` - No matches (but found general MooreToCore crash #8930)

## Analysis of Related Issues

### Issue #8332: "[MooreToCore] Support for StringType from moore to llvm dialect"
- **State**: OPEN
- **Author**: ChenXo0
- **Date**: 2025-03-20
- **Topic**: Adding StringType conversion from Moore to LLVM dialect for arcilator
- **Relation**: **Different problem** - This is a feature request/implementation discussion, not a bug report about crashes

**Comparison**:
| Aspect | Current Issue | #8332 |
|---------|--------------|--------|
| Type | Bug (crash) | Feature request |
| Trigger | `sanitizeInOut()` | Type conversion design |
| Crash Location | PortImplementation.h:177 | N/A |
| Symptom | Assertion failure | Discussion about implementation |

**Conclusion**: Not a duplicate

### Issue #8283: "[ImportVerilog] Cannot compile forward declared string type"
- **State**: OPEN
- **Date**: 2025-03-04
- **Topic**: ImportVerilog issues with string types
- **Relation**: Possibly related to string handling, but different pass

**Conclusion**: Not a duplicate (different pass and error)

## Similarity Scoring

| Issue # | Similarity Score | Reason |
|----------|------------------|--------|
| #8332 | 6.0 | Same topic (StringType), different problem (feature vs bug) |
| #8283 | 4.0 | Same topic (string types), different pass |
| #8930 | 2.0 | Same dialect (MooreToCore), different symptom |

**Threshold for duplicate**: 10.0
**Top score**: 6.0
**Is duplicate**: NO

## Recommendation

**Action**: Submit as a new issue

**Rationale**:
1. No exact match found for `sanitizeInOut` crash with string ports
2. Issue #8332 discusses StringType implementation but doesn't report a crash
3. The assertion failure location and mechanism are unique
4. This is a distinct bug that needs fixing separately

## Keywords Used
- `string`
- `StringType`
- `sanitizeInOut`
- `InOutType`
- `MooreToCore`
- `ModulePortInfo`
- `assertion`
- `crash`
