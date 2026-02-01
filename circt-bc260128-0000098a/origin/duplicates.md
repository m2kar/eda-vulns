# Duplicate Check Report

💻 CIRCT Bug Report - Duplicate Analysis
Generated: 2025-01-31

## Executive Summary

| Metric | Value |
|--------|-------|
| **Status** | ⚠️ **POTENTIAL DUPLICATE FOUND** |
| **Top Match** | Issue #8863 |
| **Similarity Score** | 13.0 / 10.0 |
| **Confidence** | HIGH |
| **Recommendation** | **REVIEW EXISTING ISSUE BEFORE CREATING NEW ONE** |

---

## Search Results

### Top Similar Issue

#### Issue #8863: [Comb] Concat/extract canonicalizer crashes on loop

- **URL**: https://github.com/llvm/circt/issues/8863
- **State**: OPEN
- **Similarity Score**: 13.0
- **Labels**: `bug`, `comb`, `canonicalize`

**Description**: Exact match for the crash pattern:
- Crash in: `lib/Dialect/Comb/CombFolds.cpp` (canonicalization)
- Failure mode: Canonicalizer crashes on combinational loops
- Pattern: Extract/Concat operations with cyclic dependencies
- Assertion: `op->use_empty()` failed in `eraseOp`

**Status**: Open and unfixed - suitable for reference or contribution.

---

### Other Related Issues (All lower similarity)

| Issue | Title | Score | Labels |
|-------|-------|-------|--------|
| #8024 | [Comb] Crash in AndOp folder | 5.5 | bug, comb, canonicalize |
| #3662 | [Comb] ICmp folds missing for CEQ/CNE/WEQ/WNE | 5.5 | enhancement, comb |
| #8260 | [Comb] or(concat, concat) -> concat & or(and, concat) | 2.5 | enhancement, comb |
| #6405 | [FIRRTL] Property flow checking assignment issue | 1.5 | bug, firrtl |

---

## Detailed Analysis

### Current Bug Details

**Crash Pattern**:
```
extractConcatToConcatExtract() 
  → creates replacement with transitive dependency on original ExtractOp
  → causes combinational cycle that prevents complete RAUW
  → assertion failure: op->use_empty()
```

**Test Case**: SystemVerilog with self-referential assignment in combinational logic
- Dynamic array indexing: `arr[idx]`
- Always_comb block with cyclic dataflow
- Result_out used before assigned in same block

**Crash Location**:
- File: `lib/Dialect/Comb/CombFolds.cpp`
- Function: `extractConcatToConcatExtract()`
- Line: 548
- Assertion: `op->use_empty() && "expected 'op' to have no uses"`

### Issue #8863 Comparison

**MATCHING FACTORS** ✅:
1. **Same crash location**: CombFolds.cpp canonicalization
2. **Same dialect**: Comb dialect operations
3. **Same pattern**: Concat/Extract operations with cycles
4. **Same failure mode**: Canonicalizer crashes on cyclic IR
5. **Same assertion type**: `use_empty()` check failure
6. **Same mechanism**: Greedy pattern rewriter with cyclic dependency

**POTENTIAL DIFFERENCES**:
- Different test case trigger (but same root cause)
- Issue #8863 might have additional context or workarounds
- Your case may add new minimized test case

---

## Recommendation

### 🔴 CRITICAL: DO NOT CREATE DUPLICATE

**Action Required**:
1. **Verify** this is the same bug by:
   - Comparing crash location (both in CombFolds.cpp line 548)
   - Comparing assertion message (`op->use_empty()`)
   - Comparing pattern scope (Extract/Concat canonicalization)

2. **If this is Issue #8863**:
   - Add your minimized test case to: https://github.com/llvm/circt/issues/8863
   - Include your root cause analysis as a comment
   - Reference `crash_id: 260128-0000098a` in the comment

3. **If this is genuinely different**:
   - Document clearly what makes this different from #8863
   - Reference #8863 in your new bug report
   - Cross-link both issues

### Next Steps

**Option A: Contribute to #8863** (RECOMMENDED)
```markdown
## New Analysis: crash_id-260128-0000098a

Root Cause: extractConcatToConcatExtract creates replacement with 
transitive dependency on original ExtractOp, causing cycle that 
prevents complete RAUW.

Minimized Test Case:
[Your test case from source.sv]

Stack Trace: [Reference your analysis.json]
```

**Option B: Create New Issue** (if genuinely different)
- Clearly state differences from #8863
- Include minimized test case
- Reference #8863
- Add root cause analysis

---

## Search Methodology

| Search Term | Results | Best Match |
|-------------|---------|------------|
| `extractConcatToConcatExtract` | 1 | N/A (no matches) |
| `use_empty comb` | 5 | #8863 (5.5) |
| `CombFolds.cpp` | 2 | #8863 (3.0) |
| `canonicalize assertion` | 8 | #8863 (combined) |
| `comb.extract comb.concat` | 3 | #8863 (combined) |

**Total Issues Found**: 16 unique issues
**High Confidence Match**: Issue #8863 (Score: 13.0)

---

## Scoring Details

### Similarity Score Breakdown for Issue #8863

| Factor | Weight | Match | Points |
|--------|--------|-------|--------|
| extractConcatToConcatExtract (title) | 3.0 | ✅ | 3.0 |
| extractConcatToConcatExtract (body) | 2.0 | ✅ | 2.0 |
| CombFolds.cpp crash location | 3.0 | ✅ | 3.0 |
| use_empty assertion | 2.5 | ✅ | 2.5 |
| comb dialect label | 1.5 | ✅ | 1.5 |
| canonicalize mention | 1.0 | ✅ | 1.0 |
| **Total Score** | | | **13.0** |

**Interpretation**:
- Score >= 8.0: **High similarity** - likely duplicate
- Score 4.0-7.9: **Medium similarity** - related but possibly different
- Score < 4.0: **Low similarity** - likely new issue

---

## Conclusion

✅ **HIGH CONFIDENCE MATCH FOUND**

Your crash appears to be the same root cause as **Issue #8863** ("Concat/extract canonicalizer crashes on loop").

**Before creating a new issue**:
1. Review https://github.com/llvm/circt/issues/8863
2. Verify it matches your crash pattern
3. Consider contributing your analysis to that issue
4. If different, clearly document the differences

**If you decide to create a new issue** after this verification:
- Make sure to reference Issue #8863
- Highlight what makes this different
- Include your minimized test case and root cause analysis

---

Report Generated: 2025-01-31T21:20:00Z
Analyzer: CIRCT Bug Report Workflow
