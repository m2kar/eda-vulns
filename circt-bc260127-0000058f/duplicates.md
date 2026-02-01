# CIRCT Duplicate Issue Check Report

## Summary

Searched GitHub llvm/circt repository for issues similar to crash bc260127-0000058f (PackedUnionType conversion failure in MooreToCore pass).

**Search Strategy:**
- Keywords: "packed union", "MooreToCore", "dyn_cast assertion", "type converter", "PackedUnionType", "sanitizeInOut"
- Scope: All open/closed issues in llvm/circt
- Total Issues Found: 14 potentially related issues
- Most Similar Score: 8/10

## Top 3 Similar Issues

### 1. Issue #8930: [MooreToCore] Crash with sqrt/floor
**Similarity Score:** 8/10  
**State:** OPEN  
**URL:** https://github.com/llvm/circt/issues/8930

**Match Criteria:**
- Same MooreToCore pass
- Same assertion failure: `dyn_cast on a non-existent value`
- Same root cause pattern: Type converter returns unsupported type → dyn_cast fails
- Same component involved: MooreToCore type converter

**Description:**  
Crash occurs when converting Moore dialect with real/math operations (sqrt, floor). Stack trace shows failure in `ConversionOpConversion::matchAndRewrite()` when trying to get bit width from null type. The issue is triggered by missing type conversion support for real/math types.

**Difference from Current Crash:**  
- Different missing conversion: real/math types vs packed union types
- Same underlying problem: Type converter incomplete

---

### 2. Issue #7535: [MooreToCore] VariableOp lowered failed
**Similarity Score:** 7/10  
**State:** OPEN  
**URL:** https://github.com/llvm/circt/issues/7535

**Match Criteria:**
- Same MooreToCore pass
- Type conversion failure on struct types passed as ports
- Stack dump shows similar pattern in type conversion
- InOutType handling issue similar to sanitizeInOut

**Description:**  
MooreToCore conversion fails when struct types are used as module ports. The issue mentions "stack dump when casting hw::InOutType" with struct types, which mirrors the null type issue in sanitizeInOut.

**Difference from Current Crash:**  
- Involves struct types instead of packed union types
- But indicates missing type support handling in same code path

---

### 3. Issue #8382: [FIRRTL] Crash with fstring type on port
**Similarity Score:** 6/10  
**State:** OPEN  
**URL:** https://github.com/llvm/circt/issues/8382

**Match Criteria:**
- Port type handling failure
- Different dialect (FIRRTL) but same pattern: unsupported type on port → assertion
- Type not handled in conversion, propagates to port processing

**Description:**  
Crash with assertion when fstring type appears on FIRRTL module port. The fstring type is not properly converted, and assertion fails when trying to verify the lowering result.

**Difference from Current Crash:**  
- Different dialect (FIRRTL vs Moore)
- Different component (LowerToHW vs MooreToCore)
- But demonstrates same systemic issue: missing type support for port types

---

## Related Issues (Lower Priority)

- **#8476, #8292, #8276, #8269**: MooreToCore type converter missing various type support (exponentiation, unsized arrays, unpacked arrays, real constants)
- **#8266**: FIRRTL integer property assertion
- **#9542, #8973, #7629, #6614**: Other MooreToCore/Moore issues (lower relevance)

## Recommendation

**Status: LIKELY_NEW**

### Rationale

While several related MooreToCore type conversion issues exist, **none specifically address PackedUnionType** support:

1. **Issue #8930** (most similar) handles different missing type conversions (sqrt/floor operations)
2. **Issue #7535** handles struct types but different context
3. **Issue #8382** is different dialect

**However:** Before creating a new issue, recommend reviewing:
- Issue #8930's patch (if any) to see the fix pattern
- Whether PackedUnionType has been added to type converter since these reports

### Next Steps

1. ✅ Review issue #8930 for similar fix patterns
2. ✅ Check if PackedUnionType is now supported in recent CIRCT commits
3. ✅ If not, create new issue with this as root cause
4. ✅ Reference issue #8930 as related/similar issue

---

## Search Details

| Query | Results |
|-------|---------|
| "packed union" | 1 issue (#6614 - low relevance) |
| "MooreToCore" | 10 issues (various MooreToCore bugs) |
| "dyn_cast assertion" | 3 issues (#8266, #8930, #8382) |
| "type converter" | 1 issue (#7535) |
| "PackedUnionType" | 0 issues |
| "sanitizeInOut" | 0 issues |

**Conclusion:** PackedUnionType not previously reported, making this likely a new/unreported issue.

---

*Report generated: 2026-01-31*  
*Crash ID: bc260127-0000058f*  
*Analysis Source: analysis.json*
