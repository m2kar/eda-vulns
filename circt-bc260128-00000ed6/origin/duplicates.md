# Duplicate Check Report

**Test Case:** 260128-00000ed6  
**Dialect:** arc  
**Crash Type:** assertion  
**Date:** 2026-02-01

## Summary

Checked CIRCT GitHub Issues for duplicates of the Arc StateType assertion crash:
- **Error:** "state type must have a known bit width; got '!llhd.ref<i1>'"
- **Crash Location:** ArcTypes.cpp.inc:108 in StateType::get()
- **Affected Component:** arc::LowerStatePass

### Search Strategy

1. **Queries Used:**
   - "arcilator assertion state type"
   - "LowerState"
   - "llhd.ref"
   - "arc state type"
   - "StateType"
   - "ArcTypes"

2. **Issues Scanned:** 20+ closed and open issues

## Findings

### Top Similar Issues

| # | Issue | Title | Similarity | Status | Reason |
|---|-------|-------|-----------|--------|--------|
| 9246 | [Arc] llhd.sig.array_slice width verification | 0.52 | closed | Involves llhd.ref type in arc context, but different error (array_slice width check) |
| 5053 | [Arc] LowerState: combinatorial cycle | 0.45 | closed | Same LowerState pass, but different issue (cycle detection) |
| 9395 | [circt-verilog][arcilator] Assertion failure | 0.48 | closed | Arcilator assertion in ConvertToArcs, related tool/pass |
| 9466 | [arcilator] llhd.constant_time lowering | 0.40 | closed | Arcilator legalization, but different operation |
| 9469 | [arcilator] Array indexing in always_ff | 0.38 | closed | Arcilator llhd legalization, similar error context |
| 9052 | [circt-verilog] llhd constant_time | 0.36 | closed | Arcilator llhd legalization failure |
| 9417 | [Arc][arcilator] bitcast data corruption | 0.35 | open | Arc dialect issue, different problem |

### Detailed Analysis

**Issue #9246 (Highest Similarity: 0.52)**
- **Component:** llhd.sig.array_slice verification
- **Error Context:** Similar type verification failure in arc/llhd context
- **Difference:** Error involves array_slice width check, not StateType bit width
- **Relevance:** Potential code path interaction, but distinct bug

**Issue #5053 (Component Match: 0.45)**
- **Component:** LowerState.cpp (same file as crash)
- **Problem:** Combinatorial cycle detection false positive
- **Relevance:** Different bug in same pass, suggests LowerState.cpp may have multiple issues

**Issues #9395, #9466, #9469 (Arcilator Legalization Pattern: 0.36-0.48)**
- **Common Pattern:** Arcilator ConvertToArcs pass legalization failures
- **Difference:** Target different operations (llhd.constant_time, not StateType)
- **Relevance:** Different root cause, likely unrelated fixes

## Recommendation

### **Classification: NEW_ISSUE** ✓

**Confidence:** HIGH (0.85+)

**Rationale:**
1. No exact duplicate found
2. Highest similarity (0.52) is insufficient for duplicate classification
3. Top match (#9246) involves different operation/error
4. All LowerState-related matches are distinct bugs
5. The specific error signature ("state type must have a known bit width; got '!llhd.ref<i1>'") does not appear in any closed issue

**Why This is Likely New:**
- **Unique Error Path:** StateType::get() validation with llhd.ref type
- **Specific Type Combination:** Assertion happens when llhd.ref<i1> is passed to StateType constructor
- **Trigger Pattern:** Occurs in inout wires with tri-state assignments (unique construct combination)

## Risk Assessment

**Risk Level:** MEDIUM

- Arc dialect StateType validation appears to have incomplete type handling
- llhd.ref type is not properly handled in computeLLVMBitWidth()
- Similar type conversion issues exist in codebase (#9246)
- Pattern suggests potential for similar failures with other llhd wrapper types

## Suggested Actions

1. **File New Issue** on llvm/circt GitHub with crash details
2. **Cross-check:** Review issue #9246 for related type handling patterns
3. **Code Review:** Check recent changes to:
   - lib/Dialect/Arc/ArcTypes.cpp (StateType::get implementation)
   - lib/Dialect/Arc/Transforms/LowerState.cpp (state lowering logic)
4. **Fix Scope:** Likely requires extending computeLLVMBitWidth() to handle llhd::RefType

## Report Generated

- **Search Date:** 2026-02-01
- **Total Issues Checked:** 20+
- **Issues with Partial Matches:** 7
- **Exact Duplicates:** 0
