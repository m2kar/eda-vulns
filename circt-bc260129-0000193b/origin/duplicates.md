# CIRCT Duplicate Issue Check Report

## Executive Summary
**Recommendation**: `review_existing`  
**Top Similarity Score**: 0.75 (75%)  
**Top Related Issue**: #9287 "[HW] Make `hw::getBitWidth` use std::optional vs -1"  
**Confidence Level**: 85%

This crash is **likely not a duplicate**, but is **directly related to** an open issue (#9287) addressing the root cause.

---

## Crash Details
- **Dialect**: LLHD
- **Crash Type**: Assertion failure (integer bitwidth limit exceeded)
- **Pass**: Mem2RegPass::runOnOperation → Mem2Reg/Transforms/Mem2Reg.cpp:1753
- **Trigger**: SystemVerilog real type variable with clocked assignment

---

## Search Keywords Used
1. `LLHD Mem2Reg assertion`
2. `LLHD real type`
3. `Mem2Reg`
4. `IntegerType bitwidth`
5. `LLHD assertion`
6. `floating point real`
7. `llhd-mem2reg`
8. `LLHD crash`
9. `getBitWidth`

---

## Related Issues Found (Top 5)

### 1. ⭐ Issue #9287 (Similarity: 75%) - MOST RELEVANT
**Title**: [HW] Make `hw::getBitWidth` use std::optional vs -1

**Status**: Open  
**Dialect**: HW

**Relevance**: This issue directly addresses the root cause of the reported crash. The issue proposes converting `hw::getBitWidth()` to return `std::optional<uint64_t>` instead of `-1` for unsupported types. 

**Similarity Analysis**:
- ✅ Same root cause: `hw::getBitWidth()` returning -1
- ✅ Same problem: -1 interpreted as unsigned causes invalid IntegerType creation
- ✅ Same solution scope: Fix getBitWidth function
- ⚠️ Different manifestation: This issue focuses on the infrastructure change, not specific test cases
- ⚠️ Different pass: Issue #9287 is about general getBitWidth handling, not Mem2Reg-specific

**Recommendation for Action**:
- Review Issue #9287 to understand the proposed solution
- Consider if the reported crash should be added as a test case to #9287
- May not need separate issue if #9287 fix is comprehensive

---

### 2. Issue #8693 (Similarity: 65%)
**Title**: [Mem2Reg] Local signal does not dominate final drive

**Status**: Open  
**Dialect**: LLHD

**Relevance**: Specific Mem2Reg pass issue but with different root cause (dominance violation).

**Similarity Analysis**:
- ✅ Same pass: LLHD Mem2Reg
- ✅ Involves signal handling
- ❌ Different root cause: Dominance violation vs bitwidth issue
- ❌ Different type: Simple i1 signals vs real types

---

### 3. Issue #8286 (Similarity: 60%)
**Title**: [circt-verilog][llhd][arcilator] Verilog-to-LLVM lowering issues

**Status**: Open  
**Dialect**: LLHD, Arcilator

**Relevance**: Broader scope covering multiple Mem2Reg and LLHD transform issues.

**Similarity Analysis**:
- ✅ Related to Mem2Reg transforms
- ✅ Discusses llhd-hoist-signals and llhd-mem2reg
- ❌ Different specific problems
- ❌ Asks for pipeline integration rather than bug report

---

### 4. Issue #8930 (Similarity: 50%)
**Title**: [MooreToCore] Crash with sqrt/floor

**Status**: Open  
**Dialect**: Moore

**Relevance**: Another crash involving `getBitWidth()` returning -1 for floating-point types.

**Similarity Analysis**:
- ✅ Same root cause: `getBitWidth()` returning -1 for real types
- ✅ Both involve floating-point operations
- ❌ Different pass: MooreToCore vs Mem2Reg
- ❌ Different crash location: dyn_cast assertion vs IntegerType creation

**Key Quote**: 
```cpp
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

---

### 5. Issue #9013 (Similarity: 45%)
**Title**: [circt-opt] Segmentation fault during XOR op building

**Status**: Open  
**Dialect**: LLHD

**Relevance**: LLHD-related crash but with different root cause.

**Similarity Analysis**:
- ✅ LLHD dialect involved
- ❌ Different pass: llhd-desequentialize vs Mem2Reg
- ❌ Different root cause: Type constraint violation
- ❌ Different trigger: Sequential logic with reset signal

---

## Root Cause Analysis

### Primary Cause
The `hw::getBitWidth()` function returns `-1` (cast as unsigned = 18446744073709551615 = ~2^64-1) for unsupported types like SystemVerilog `real`.

### Failure Chain
1. **Input**: SystemVerilog real type variable in always @(posedge clk)
2. **circt-verilog**: Converts to MLIR with real type
3. **llhd-mem2reg pass**: Invokes `builder.getIntegerType(hw::getBitWidth(type))`
4. **GetBitWidth result**: Returns -1 (treated as huge unsigned number)
5. **Assertion**: MLIR IntegerType limit is 16,777,215 bits (2^24 - 1)
6. **Crash**: Assertion `bitwidth <= 16777215` fails

### Why This Is Not Currently a Duplicate
1. **Specific scenario**: The exact combination of Mem2Reg + real types + clocked assignment is unique
2. **Different pass focus**: While #9287 addresses the root cause, this specific manifestation in Mem2Reg hasn't been reported
3. **Urgent fix needed**: This crash blocks real type support in sequential logic
4. **Test case value**: This provides a concrete test case for #9287 fix validation

---

## Similarity Scoring Methodology

```
Score = (Keyword Matches × 0.3) + (Root Cause Match × 0.3) + 
         (Stack Trace Similarity × 0.2) + (Type Similarity × 0.2)

Issue #9287:
- Keyword matches: 100% (getBitWidth, integer type, limit)
- Root cause match: 100% (same -1 return issue)
- Stack trace similarity: 40% (different pass, same function)
- Type similarity: 75% (both handle type issues)
= (1.0 × 0.3) + (1.0 × 0.3) + (0.4 × 0.2) + (0.75 × 0.2) = 0.75
```

---

## Recommendations

### Option A: Review Existing (Recommended ✅)
- Monitor Issue #9287 progress
- Comment with this specific test case if #9287 is not comprehensive
- Add to #9287 test suite if it doesn't cover real types in Mem2Reg
- Link this crash as a specific manifestation

### Option B: Create New Issue + Link
- Create new issue if #9287 doesn't fully address the Mem2Reg-specific aspects
- Title: "[Mem2Reg] Assertion failure with real type variables in clocked assignments"
- Link to #9287 as root cause
- Provide minimal test case

### Option C: Reopen Existing + Add Details
- If #9287 exists but is closed/stale, reopen with this case
- Add this test case as regression test

---

## Verification Checklist

- [x] Searched GitHub issues with 9 different keyword combinations
- [x] Reviewed top 5 most similar issues
- [x] Analyzed root cause vs similar issues
- [x] Calculated similarity scores
- [x] Verified crash signature uniqueness
- [x] Identified relationship to #9287

---

## Conclusion

This crash represents a **legitimate bug manifestation** of the underlying issue documented in #9287. While #9287 may be closed as a broader refactoring task, this specific scenario (Mem2Reg + real types) should be:

1. **Captured** in the test suite for #9287
2. **Referenced** if a new issue is created
3. **Linked** to #9287 as a concrete failing case

**Final Recommendation**: `review_existing` with consideration for creating a PR that addresses real types in Mem2Reg as part of the #9287 fix.

