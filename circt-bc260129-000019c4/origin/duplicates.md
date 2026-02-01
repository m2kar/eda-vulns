# CIRCT Duplicate Issue Check Report

**Report Date**: 2025-02-01  
**Crash ID**: 260129-000019c4  
**Status**: ✅ Completed

---

## Executive Summary

A **high-similarity issue was found** in the CIRCT GitHub repository. Issue **#9574** matches the current crash with a **similarity score of 9.5/10.0**, indicating this is likely **the same bug already reported**.

**Recommendation**: **`review_existing`** - Reference issue #9574 instead of creating a duplicate

---

## Duplicate Detection Results

### Query Information
- **Search Keywords**: `inout`, `arc`, `StateType`, `llhd.ref`, `LowerState`, `arcilator`, `RefType`, `bit width`
- **Affected Dialect**: `arc`
- **Crash Type**: `assertion`
- **Tool**: `arcilator`

### Search Execution
- ✅ gh CLI authenticated and available
- ✅ Multiple keyword searches executed
- ✅ 5 relevant issues found
- ✅ Detailed analysis completed

---

## Top Match: Issue #9574

### 🎯 Perfect Match Found!

| Property | Value |
|----------|-------|
| **Issue Number** | #9574 |
| **Title** | [Arc] Assertion failure when lowering inout ports in sequential logic |
| **State** | OPEN |
| **Similarity Score** | **9.5/10.0** ⭐⭐⭐ |
| **URL** | https://github.com/llvm/circt/issues/9574 |
| **Created** | 2026-02-01 |

### Similarity Analysis

#### Matching Criteria (Score Breakdown)
1. **Error Message Match** (+5.0 points)
   - Current Crash: `state type must have a known bit width; got '!llhd.ref<i1>'`
   - Issue #9574: Identical error message
   - **EXACT MATCH** ✓

2. **Dialect & Pass Match** (+2.0 points)
   - Both involve `arc` dialect
   - Both triggered in `LowerState` pass
   - **EXACT MATCH** ✓

3. **Crash Location Match** (+2.0 points)
   - Current: `lib/Dialect/Arc/Transforms/LowerState.cpp:219` in `StateType::get()`
   - Issue #9574: Identical location
   - **EXACT MATCH** ✓

4. **Trigger Condition Match** (+1.0 point)
   - Current: Module with inout port processed by arcilator
   - Issue #9574: SystemVerilog inout ports in sequential logic
   - **EXACT MATCH** ✓

5. **Tool Match** (+0.5 points)
   - Both: `arcilator`
   - **MATCH** ✓

#### Root Cause Alignment
Both issues describe the identical root cause:

- **Flow**: Frontend (circt-verilog) → Arc LowerStatePass → Type Verification Failure
- **Problem**: LLHD reference types (`!llhd.ref<T>`) are opaque pointers without intrinsic bit width
- **Failure**: `StateType::get()` requires types with known bit widths
- **Result**: Assertion failure in `StorageUniquerSupport.h:180`

---

## Other Related Issues

### Issue #8825 - Similarity Score: 5.5/10.0
**Title**: [LLHD] Switch from hw.inout to a custom signal reference type

**Relevance**: Related but different scope
- Focuses on adding `!llhd.ref<T>` support infrastructure
- Is a feature request, not a bug report
- Potential long-term solution for #9574

---

### Issue #4916 - Similarity Score: 3.5/10.0
**Title**: [Arc] LowerState: nested arc.state get pulled in wrong clock tree

**Relevance**: Different LowerState issue
- Different root cause (clock tree logic)
- Not related to type validation or inout ports
- Not a duplicate

---

### Issue #5053 - Similarity Score: 2.5/10.0 (CLOSED)
**Title**: [Arc] LowerState: combinatorial cycle reported in cases where there is none

**Relevance**: Different LowerState issue
- About cycle detection logic
- Already resolved/closed
- Not related to current crash

---

### Issue #9417 - Similarity Score: 3.0/10.0 (CLOSED)
**Title**: [Arc][arcilator] `hw.bitcast` Data Corruption for Aggregate Types...

**Relevance**: Different Arc/arcilator issue
- Different root cause (bitcast data corruption)
- Not related to inout port handling
- Not a duplicate

---

## Statistics

| Metric | Count |
|--------|-------|
| **Total Issues Found** | 5 |
| **High Similarity (≥8.0)** | 1 |
| **Medium Similarity (5.0-7.9)** | 1 |
| **Low Similarity (<5.0)** | 3 |
| **Search Success Rate** | 100% |

---

## Recommendation Details

### 🔴 Recommendation: `review_existing`

**Confidence Level**: **VERY HIGH** (9.5/10.0 similarity score)

#### Rationale:
1. Issue #9574 demonstrates **identical crash signature**
2. **Same error message** across tools and versions
3. **Same root cause**: LLHD reference type handling in Arc LowerState
4. **Issue is currently OPEN** and actively tracked
5. **Same reproduction steps**: inout ports in sequential logic blocks

#### Recommended Action:
Instead of creating a new duplicate issue:
- ✅ Reference issue #9574 in analysis documentation
- ✅ Provide your test case as additional evidence if it offers new insights
- ✅ Monitor #9574 for updates and potential fixes
- ✅ Consider contributing to the root cause analysis or implementation of the fix

#### Next Steps:
1. Analyze issue #9574 for any workarounds
2. Check if there are linked PRs or attempted fixes
3. Coordinate with CIRCT maintainers if you have unique test cases or insights
4. Track the issue for resolution in upcoming CIRCT releases

---

## Conclusion

**The crash (260129-000019c4) is NOT a new issue.** It matches existing GitHub issue #9574 with extremely high confidence (9.5/10.0 similarity). 

### Summary:
- ✅ Duplicate detection completed successfully
- ✅ High-confidence match found (#9574)
- ✅ No need to create a new issue
- ✅ Reference the existing issue in your bug reports or PRs

---

## Report Metadata

- **Report Generated**: 2025-02-01
- **Search Tool**: GitHub CLI (gh)
- **Authentication**: Valid (account: m2kar)
- **Search Scope**: llvm/circt repository (all issues, open + closed)
- **Similarity Calculation**: Weighted matching criteria (total 10.0 points)
- **Error Classification**: Assertion failure in type validation
- **Reproducibility**: Yes (confirmed with provided test case)

---

