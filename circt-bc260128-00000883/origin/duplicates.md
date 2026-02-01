# CIRCT Duplicate Issue Analysis
## Test Case: 260128-00000883 (Packed Union Port Type Crash)

### Crash Summary
- **Signature**: `dyn_cast on a non-existent value` 
- **Location**: `PortImplementation.h:177` in `sanitizeInOut()`
- **Pass**: MooreToCore
- **Root Cause**: Type converter returns null for packed union port type, leading to assertion failure during InOutType dyn_cast

---

## Search Results: 4 Issues Found

### ⭐ Top Match: Issue #8930 [MOST SIMILAR - Score: 0.85]
**Title**: [MooreToCore] Crash with sqrt/floor  
**State**: OPEN  
**Link**: https://github.com/llvm/circt/issues/8930

**Crash Details**:
- Same dyn_cast assertion: `dyn_cast on a non-existent value`
- Same pass: MooreToCore conversion
- Same error pattern: Type conversion fails, returns null/invalid type
- Function: `ConversionOpConversion::matchAndRewrite()`

**Why Similar**:
- Identical assertion failure signature
- Same MooreToCore pass context
- Root cause pattern: failed type converter returning unsupported type
- Both involve conversion operation failures

**Why Might Be Different**:
- Triggers on ConversionOp (sqrt/floor expressions) vs module port declaration
- Different type construct (real/numerical vs packed union)

**Recommendation**: ⚠️ **LIKELY SAME BUG CLASS** - Both fail due to insufficient type converter coverage. Should investigate if fix for one resolves the other.

---

### 🔶 Second Match: Issue #7535 [LIKELY DUPLICATE - Score: 0.78]
**Title**: [MooreToCore] VariableOp lowered failed  
**State**: OPEN  
**Link**: https://github.com/llvm/circt/issues/7535

**Crash Details**:
- MooreToCore pass failure during InOutType casting
- Stack dump: DialectConversion.cpp line mentions InOutType issues
- Involves type conversion of structured types to HW dialect

**Why Similar**:
- Same MooreToCore pass (stronger correlation)
- Involves InOutType casting (exact same function failing: `sanitizeInOut`)
- Both fail to handle certain types during port/variable lowering
- Both involve struct/packed types on ports

**Why Might Be Different**:
- Triggers on VariableOp instead of module port declaration
- Uses struct instead of union (same family, different construct)
- Stack trace shows different path, but same end function

**Recommendation**: ✅ **LIKELY DUPLICATE** - Same MooreToCore+InOutType combination. Different triggering construct but same underlying type validation issue. Consider marking as duplicate once #260128-00000883 is fixed.

---

### 🟡 Third Match: Issue #8382 [SIMILAR PATTERN - Score: 0.65]
**Title**: [FIRRTL] Crash with fstring type on port  
**State**: OPEN  
**Link**: https://github.com/llvm/circt/issues/8382

**Crash Details**:
- dyn_cast assertion on non-existent value
- Type validation failure: getBaseType() returns invalid type
- Port-related (fstring type on module port)

**Why Similar**:
- Same dyn_cast assertion pattern
- Port-related type conversion failure
- Type validation missing before cast

**Why Different**:
- Different dialect: FIRRTL→HW instead of Moore→HW
- Different function: getBaseType() vs sanitizeInOut()
- Different type system, not MooreToCore pass

**Recommendation**: ⚠️ **PATTERN MATCH** - Similar vulnerability pattern but different dialects. Fix for #260128-00000883 might not help, but similar solution approach (type validation before casting) could apply.

---

### ⚪ Fourth Match: Issue #8266 [UNRELATED - Score: 0.35]
**Title**: [FIRRTL] Integer Property folders assert in getAPSInt  
**State**: OPEN  
**Link**: https://github.com/llvm/circt/issues/8266

**Why Not Similar**:
- Different dialect (FIRRTL only, not Moore)
- Different assertion (APSInt about signless integers vs dyn_cast)
- Different root cause (attribute type vs type conversion)
- Different function stack

**Recommendation**: ❌ **NOT RELATED** - Skip this issue.

---

## Final Assessment

### Recommendation: **likely_new**

**Summary**:
- ✅ **#7535** is likely the same bug (MooreToCore + InOutType, different trigger)
- ⚠️ **#8930** has identical error but different trigger (might be class of bugs)
- ⚠️ **#8382** shows similar vulnerability pattern but in different dialect
- ❌ **#8266** is unrelated

### Action Items:
1. **Before filing new issue**: Compare with #7535 in detail. If fix applies to both, mark as duplicate.
2. **Cross-check with #8930**: Ensure type converter handles all type cases (not just real/int/float).
3. **Consider broader fix**: Type validation pattern should apply to all type conversions in MooreToCore, not just packed unions.

### Top Scores:
- **Highest Similarity**: #8930 (0.85) - Same error, different construct
- **Most Likely Duplicate**: #7535 (0.78) - Same pass, same function failing
- **Issue to Check**: #8382 (0.65) - Pattern-based, different dialect

---

## Metadata
- **Search Queries Executed**: 8
- **Successful Searches**: 6
- **Total Issues Found**: 4
- **Analysis Date**: 2026-01-31
- **GitHub Repo**: llvm/circt
