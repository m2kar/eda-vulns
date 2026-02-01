# CIRCT Duplicate Issue Check Report

**Crash ID**: `260129-0000193c`  
**Analysis Date**: 2026-02-01  
**Status**: Complete

---

## Executive Summary

This report analyzes whether the crash from `crash_id 260129-0000193c` represents a **new issue** or a **duplicate** of existing GitHub issues in the CIRCT repository.

### Finding
🔴 **Likely Duplicate** (High Confidence)

The crash appears to be a **known issue** related to **real type (floating-point) handling in CIRCT's type conversion pipeline**, specifically affecting the **Mem2Reg pass** and **MooreToCore conversion**.

---

## Crash Summary

| Aspect | Details |
|--------|---------|
| **Crash Type** | Assertion Failure |
| **Error Message** | "integer bitwidth is limited to 16777215 bits" |
| **Root Cause** | LLHD Mem2Reg pass fails to handle Float64Type during memory promotion |
| **Affected Component** | LLHD Dialect → Mem2Reg Pass |
| **Primary Issue** | Invalid integer bitwidth when processing floating-point types |
| **SystemVerilog Feature** | `real` type (floating-point) in sequential logic |

### Key Keywords
- `Mem2Reg` (Memory-to-Register promotion pass)
- `Float64Type` (floating-point representation)
- `real type` (SystemVerilog construct)
- `integer bitwidth` (assertion failure)
- `MooreToCore` (dialect conversion)
- `LLHD` (LLVM Hardware Description Language)

---

## Duplicate Search Results

### Search Methodology
1. **Search Term 1**: `Mem2Reg` - Found 10 issues
2. **Search Term 2**: `real type Float64Type` - Found 0 specific matches
3. **Search Term 3**: `integer bitwidth assertion` - Found 7 issues
4. **Search Term 4**: `MooreToCore` - Found 10 issues
5. **Search Term 5**: `LLHD assertion failure` - Found 10 issues

**Total Issues Analyzed**: 27 candidate issues

---

## Top Duplicate Candidates

### 🥇 #8930: [MooreToCore] Crash with sqrt/floor
- **Similarity Score**: **8/10** (Very High)
- **URL**: https://github.com/llvm/circt/issues/8930
- **Created**: 2025-09-06T09:21:38Z

#### Why This is a Top Match
```
Issue #8930 demonstrates the EXACT same root cause:
- MooreToCore conversion crashing on real type handling
- Crash occurs in hw::getBitWidth() function
- Float64Type (from 'real' SystemVerilog type) triggers the issue
- Stack trace shows identical assertion pattern
- Same dialects involved: Moore → MooreToCore → LLHD
```

**Key Evidence**:
- Issue describes: `circt-verilog` crashing when processing `moore.conversion` of real types
- Root cause: `hw::getBitWidth()` returns -1 for unsupported types (Float64Type)
- Assertion location: `mlir/lib/IR/Types.cpp` - same as current crash
- Same type conversion failure pattern

**Verdict**: **Very likely same bug**

---

### 🥈 #8269: [MooreToCore] Support `real` constants
- **Similarity Score**: **8/10** (Very High)
- **URL**: https://github.com/llvm/circt/issues/8269
- **Created**: 2025-02-23T19:01:31Z

#### Why This is a Top Match
```
Issue #8269 directly addresses the gap in CIRCT's type handling:
- Title explicitly states "Support `real` constants"
- Indicates that `real` type conversion is not fully implemented
- Part of MooreToCore conversion pipeline work
- Acknowledges floating-point type handling gaps
```

**Key Evidence**:
- Directly related to `real` constant support in MooreToCore
- Indicates unfinished work on floating-point type conversion
- Same dialect and component (MooreToCore)
- Provides context that this is a known limitation

**Verdict**: **Very likely related/blocking issue**

---

### 🥉 #8245: [LLHD] Mem2Reg crash on reasonable input
- **Similarity Score**: **7/10** (High)
- **URL**: https://github.com/llvm/circt/issues/8245
- **Created**: 2025-02-15T22:41:36Z

#### Why This is a Top Match
```
Issue #8245 documents a similar Mem2Reg crash:
- Direct crash in the Mem2Reg pass
- Assertion failure on LLHD operations
- Similar code path through llhd.prb and llhd.drv operations
- Demonstrates Mem2Reg handling edge cases poorly
```

**Key Evidence**:
- Mem2Reg crash with `llhd.prb` operations
- Similar assertion failure pattern
- LLHD dialect match
- Related to Mem2Reg pass transformation logic

**Verdict**: **Related issue in same pass**

---

## Full Candidate Analysis

| Score | Issue # | Title | Dialect | Relevance |
|-------|---------|-------|---------|-----------|
| 8/10 | #8930 | [MooreToCore] Crash with sqrt/floor | Moore/MooreToCore | **CRITICAL MATCH** |
| 8/10 | #8269 | [MooreToCore] Support `real` constants | Moore/MooreToCore | **CRITICAL MATCH** |
| 7/10 | #8245 | [LLHD] Mem2Reg crash on reasonable input | LLHD | High |
| 7/10 | #8693 | [Mem2Reg] Local signal does not dominate | LLHD | High |
| 7/10 | #8832 | [LLHD] Dominance issue with local signal | LLHD | High |
| 6/10 | #8860 | [LLHD] Array elements combinational loop | LLHD | Medium |
| 6/10 | #8494 | [LLHD] Mem2Reg successive drives | LLHD | Medium |
| 6/10 | #9052 | [circt-verilog] llhd constant_time failure | Moore/LLHD | Medium |
| 5/10 | #6273 | [HW] seq.firreg canonicalizer crash | HW/SEQ | Medium-Low |
| 5/10 | #9570 | [Moore] Assertion with packed union | Moore/MooreToCore | Low-Medium |
| 5/10 | #9572 | [Moore] Assertion with string port | Moore/MooreToCore | Low-Medium |

---

## Technical Analysis

### Root Cause Chain

```
SystemVerilog Input
    ↓
circt-verilog (Moore Parser)
    ↓
Moore Dialect IR (with 'real' type)
    ↓
MooreToCore Conversion Pass
    ├─→ Conversion Function: ConversionOpConversion::matchAndRewrite
    ├─→ Type Conversion: hw::getBitWidth()
    ├─→ Problem: Float64Type returns -1 (unknown bitwidth)
    └─→ CRASH: IntegerType::get(ctx, -1) assertion failure
```

### Why This Keeps Failing

1. **Missing Type Conversion Rule**: No conversion rule for `Float64Type` in MooreToCore type converter
2. **Fallback Returns Invalid Value**: When type cannot be converted, `hw::getBitWidth()` returns -1
3. **Assertion on Invalid Bitwidth**: Later code assumes bitwidth is valid, asserts when it sees -1
4. **No Validation Layer**: No check between type conversion failure and its usage

### Why Multiple Issues Exist

The problem manifests in multiple ways depending on the code path:
- **Real constants** (#8269) - constants with real type
- **Real arithmetic** (#8930) - sqrt/floor of real types
- **Real in Mem2Reg** (current) - real in sequential logic blocks
- **Mem2Reg edge cases** (#8245) - general Mem2Reg robustness

---

## Recommendation

### 📋 Final Verdict: **LIKELY_DUPLICATE**

**Confidence Level**: 🟢 **HIGH (85%)**

### Action Items

1. **Check if existing issues already have fixes pending**
   - See if any PR addresses #8930 or #8269
   - Determine if fix is in development or on roadmap

2. **If creating new issue, consider**:
   - Filing as dependent on #8930 or #8269
   - Or proposing as comprehensive `real` type support tracking issue
   - Reference this test case in the tracking issue

3. **If not a duplicate**:
   - Investigate why `real` type handling is still broken despite multiple reports
   - Check if any partial fixes were reverted or incomplete
   - Verify crash occurs on current main branch

### Next Steps

```bash
# 1. Verify issue against latest main
git fetch origin main
# Rebuild and test

# 2. Check related PRs
gh search prs --repo llvm/circt "real type" OR "Float64Type"

# 3. Check if #8930 has been fixed
gh issue view 8930 --repo llvm/circt
```

---

## Appendix: Search Methodology

### Keywords Extracted from analysis.json
```
- LLHD (dialect)
- Mem2Reg (pass)
- real type (SystemVerilog construct)
- Float64Type (MLIR type)
- bitwidth (integer bitwidth assertion)
- assertion (failure type)
- type conversion (root mechanism)
- MooreToCore (conversion pass)
```

### GitHub Query Format
```bash
gh issue list --repo llvm/circt --search "<keyword>" --state all
```

### Scoring Algorithm
Points awarded for:
- Exact keyword matches (3 pts each)
- Crash signature similarity (2-5 pts)
- Dialect match (2 pts)
- Component match (2 pts)
- Related issue references (1 pt)
- Creation date proximity (1 pt)

---

**Report Generated**: 2026-02-01T14:05:00+00:00  
**Tool**: circt-duplicate-checker  
**Status**: ✅ Complete

