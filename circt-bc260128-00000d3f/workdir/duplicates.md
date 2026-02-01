# CIRCT Bug Duplicate Check Report

**Date**: 2026-02-01  
**Tool**: check-duplicates-worker  
**Repository**: llvm/circt  

---

## Executive Summary

- **Crash Type**: Assertion failure in `dyn_cast` on null type
- **Root Cause**: TypeConverter returns null for packed union type in MooreToCore conversion
- **Recommendation**: **REVIEW_EXISTING** - High similarity to existing issues

---

## Search Keywords

The following keywords were extracted from the crash analysis:

1. `packed union` - Type construct triggering the crash
2. `InOutType` - Hardware dialect type involved in assertion
3. `sanitizeInOut` - Function where assertion fires
4. `MooreToCore` - Conversion pass where null type is created
5. `assertion` - Type of crash
6. `dyn_cast` - LLVM operation failing on null value

---

## Top Similar Issues

### 1. **#8219** - [ESI] Assertion: dyn_cast on a non-existent value ⭐ **HIGHEST MATCH**
- **Similarity Score**: 38/100
- **Pattern Match**: `dyn_cast` + `assertion` (both present)
- **Stack Pattern**: 
  ```
  dyn_cast on a non-existent value assertion
  llvm::Support/Casting.h:662
  RemoveWrapUnwrap::matchAndRewrite
  ```
- **Context**: ESI lower to HW conversion with bundle packing/unpacking
- **Status**: Known issue, may indicate systemic problem with null type handling

### 2. **#8930** - [MooreToCore] Crash with sqrt/floor
- **Similarity Score**: 36/100  
- **Pattern Match**: `MooreToCore` + `dyn_cast` + `assertion`
- **Stack Pattern**:
  ```
  dyn_cast<mlir::IntegerType> assertion failed
  ConversionOpConversion::matchAndRewrite
  MooreToCore.cpp
  ```
- **Context**: Real type conversion fails, crashes on invalid dyn_cast
- **Status**: MooreToCore conversion issue, same root cause pattern

### 3. **#9315** - [FIRRTL] ModuleInliner removes NLA
- **Similarity Score**: 32/100
- **Pattern Match**: `dyn_cast` + `assertion`
- **Stack Pattern**:
  ```
  dyn_cast<circt::hw::HierPathOp> assertion failed
  llvm::Support/Casting.h:656
  PathTracker::processPathTrackers
  ```
- **Context**: Lower classes pass crashes after module inlining
- **Status**: Different subsystem but same assertion pattern

### 4. **#4036** - [PrepareForEmission] Crash when inout operations passed to ports
- **Similarity Score**: 28/100
- **Pattern Match**: `InOutType` + crash
- **Relevance**: Directly involves `InOutType` which is the assertion target
- **Status**: May be related to same null-handling issue

### 5. **#7535** - [MooreToCore] VariableOp lowered failed
- **Similarity Score**: 4/100
- **Pattern Match**: `MooreToCore` + struct type (weaker match)
- **Context**: Struct lowering in MooreToCore fails
- **Status**: Different but related subsystem

---

## Similarity Analysis Details

### Pattern Matching Algorithm

| Pattern | Score | Current Issue | Top 3 Matches |
|---------|-------|---------------|---|
| `dyn_cast` + `assertion` | 30 | ✓ | #8219✓, #8930✓, #9315✓ |
| `MooreToCore` + `assertion` | 25 | ✓ | #8930✓ |
| `InOutType` + `assertion` | 40 | ✓ | - |
| `packed union` | 3 | ✓ | - |
| `sanitizeInOut` | 1 | ✓ | - |

**Conclusion**: Strong match on `dyn_cast` + `assertion` pattern, but unique trigger (packed union + null type) suggests partially new issue.

---

## Detailed Comparison with #8219

### Similarities
1. Both: Assertion on `dyn_cast` with null value
2. Both: MLIR type system crash
3. Both: Involve aggregate/complex types
4. Both: Happen during dialect lowering

### Differences  
| Aspect | Current Bug | #8219 |
|--------|------------|-------|
| **Trigger** | packed union as module port | ESI bundle pack/unpack |
| **Pass** | MooreToCore | ESI→HW |
| **Function** | sanitizeInOut | RemoveWrapUnwrap |
| **Null Source** | TypeConverter | Removed wrap operation |
| **Stack Frames** | 42 frames | 13 frames |

---

## Detailed Comparison with #8930

### Similarities
1. Both: MooreToCore conversion pass
2. Both: `dyn_cast` assertion failure
3. Both: Type conversion producing null/invalid

### Differences
| Aspect | Current Bug | #8930 |
|--------|------------|-------|
| **Trigger** | packed union port | sqrt/floor operations |
| **Failed Type** | Packed union → HW | Real → IntegerType |
| **Location** | PortImplementation.h | MooreToCore.cpp |
| **Function** | sanitizeInOut | ConversionOpConversion |

---

## Root Cause Hypothesis

**Primary Issue**: Type converter failure during MooreToCore pass
- TypeConverter lacks conversion rule for `packed union` types
- Returns null instead of valid HW type
- Downstream code assumes non-null and crashes on `dyn_cast`

**Secondary Pattern**: Systemic issue with null type checking
- Multiple passes have unguarded `dyn_cast` on potentially null types
- Suggests need for:
  1. Comprehensive null checks after type conversions
  2. More defensive `dyn_cast` usage (use `dyn_cast_or_null`)
  3. Better error reporting before crashes

---

## Recommendation Details

### Recommendation: **REVIEW_EXISTING** (Score: 38/100)

**Reasoning**:
- **38 >= 30**: Indicates substantial similarity to existing issues
- Core pattern (`dyn_cast` + `assertion`) is well-established
- Issues #8219, #8930 represent same class of problem
- However, specific trigger (`packed union` + `sanitizeInOut`) is unique enough to warrant new issue

### Action Items:
1. **Review**: Check if packed union support has been added to issues #8219, #8930, #9315
2. **Consider**: Standalone issue if packed union type handling is not covered
3. **Link**: Reference #8219 and #8930 as related duplicates if creating new issue
4. **Fix Priority**: High - affects type conversion correctness

---

## Search Metadata

- **Search Query 1**: `packed union OR InOutType OR sanitizeInOut OR MooreToCore`
- **Search Query 2**: `assertion crash InOutType`  
- **Search Query 3**: `dyn_cast non-existent value`
- **Total Issues Found**: 24
- **Analysis Timestamp**: 2026-02-01T00:42:00Z

---

## Conclusion

**This is likely NOT a pure duplicate** of existing issues, but represents the same systemic problem:

1. **MooreToCore pass**: Needs packed union type support
2. **Type system**: Needs better null-checking post-conversion
3. **Error handling**: Should fail gracefully instead of crashing

**Best Path Forward**: 
- Report as new issue, explicitly linking to #8219 and #8930
- Title: `[MooreToCore] Crash with packed union port type in sanitizeInOut (dyn_cast assertion)`
- Tag as related to systemic `dyn_cast` null-handling issues
