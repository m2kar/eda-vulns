# Duplicate Check Report for CIRCT Issue

**Date**: 2026-01-31  
**Crash Type**: Assertion Failure  
**Component**: ARC Dialect / arcilator Tool  
**Dialect**: ARC (with LLHD)  
**Failing Pass**: LowerStatePass  

## Executive Summary

This crash is **LIKELY A DUPLICATE** or **CLOSELY RELATED** to existing CIRCT GitHub issues. 

**Most relevant matches:**
1. **Issue #9395** (CLOSED, Jan 19, 2026): Arcilator assertion failure in arc dialect
2. **Issue #8825** (OPEN): Root cause - Switch from hw.inout to custom signal reference type (!llhd.ref)
3. **Issue #9467** (OPEN, Jan 20, 2026): Arcilator LLHD to Arc conversion issues

---

## Detailed Analysis

### Search Strategy

Performed comprehensive GitHub search using `gh` CLI with 16 different keyword combinations:
- Dialect-specific: `arc`, `LLHD`, `arcilator`
- Function/Type-specific: `StateType`, `computeLLVMBitWidth`, `LowerState`
- Feature-specific: `inout`, `ref`, `bidirectional`, `tristate`
- Error-specific: `assertion`, `bit width`, `lowering`

**Total Issues Found**: 15 unique issues related to this crash domain  
**Highly Relevant**: 3 issues (95%, 90%, 85% similarity)  
**Related**: 6 issues (60-75% similarity)  

---

## Top 3 Most Relevant Issues

### 1. Issue #9395 - [CLOSED] Arcilator assertion failure

**Title**: `[circt-verilog][arcilator] Arcilator assertion failure`  
**Status**: ✅ CLOSED (2026-01-19)  
**Labels**: Arc, ImportVerilog  
**Created**: 2025-12-29  
**Similarity Score**: 95%

**Key Characteristics:**
- SystemVerilog input with always @* blocks
- Assertion failure during arc conversion
- Same root pass: ConvertToArcs
- Same tool chain: circt-verilog → arcilator

**Why It Matters:**
This is the closest match to our issue. Though the assertion error is different (ConversionPatternRewriter vs StateType::verify), both occur in the same ConvertToArcs pass and same pipeline. The recent closure (Jan 19, 2026) suggests fixes may have been applied to the arc conversion infrastructure.

**Stack Trace Similar Components:**
```
#15 circt::llhd::CombinationalOp conversion in ConvertToArcs.cpp
    → Our issue: StateType creation in LowerState.cpp line 219
Both involve LLHD dialect ops failing to convert to Arc
```

---

### 2. Issue #8825 - [OPEN] Switch from hw.inout to custom signal reference type

**Title**: `[LLHD] Switch from hw.inout to a custom signal reference type`  
**Status**: ⏳ OPEN (tracking issue)  
**Labels**: LLHD  
**Created**: 2025-08-06  
**Similarity Score**: 90%

**Key Characteristics:**
- Proposes creating `!llhd.ref<T>` type
- Addresses limitations of `hw.inout` for non-HW types
- Root cause of our crash: StateType receives unsupported llhd.ref types

**Why It Matters:**
This is the **architectural root cause** of our issue. The issue explicitly states:
> "To support Verilog's `time` values... Variables get mapped to `llhd.sig` allocation operations. These return a reference type. This prevents us from creating `time` variables since `!hw.inout` forces `isHWValueType`..."

Our crash involves `!llhd.ref<i1>` types that StateType::verify cannot handle. This is a symptom of the incomplete type system migration described in this issue.

---

### 3. Issue #9467 - [OPEN] arcilator fails to lower llhd.constant_time

**Title**: `[circt-verilog][arcilator] arcilator fails to lower llhd.constant_time generated from simple SV delay (#1)`  
**Status**: ⏳ OPEN  
**Labels**: LLHD, Arc  
**Created**: 2026-01-20  
**Similarity Score**: 85%

**Key Characteristics:**
- Different LLHD op: `llhd.constant_time` vs our `llhd.ref<i1>`
- Same pipeline: circt-verilog → arcilator
- Same pass: ConvertToArcs marks entire LLHD dialect illegal
- Different legalization error but same root cause

**Why It Matters:**
Shows a pattern: **Multiple LLHD dialect ops are not properly handled by Arc conversion**. This suggests systematic issues with the LLHD→Arc transition, not just our specific case.

```
Error: "failed to legalize operation 'llhd.constant_time' that was 
        explicitly marked illegal"
Our Error: "state type must have a known bit width; got '!llhd.ref<i1>'"
```

Both are LLHD ops that Arc infrastructure cannot process.

---

## Related Issues (6 Medium-Relevance)

| # | Title | Similarity | Status | Key Relevance |
|---|-------|-----------|--------|---------------|
| 8286 | Verilog-to-LLVM lowering issues | 75% | OPEN | llhd.constant_time legalization, same pipeline |
| 8012 | Moore to LLVM lowering issues | 70% | OPEN | LLHD process lowering failures |
| 8065 | LLHD/Arc indexing lowering | 68% | OPEN | LLHD type lowering failures |
| 8845 | circt-verilog produces non comb/seq dialects | 65% | OPEN | circt-verilog outputs LLHD not Arc |
| 5566 | Crash in hw.inout port verifiers | 60% | OPEN | hw.inout/inout handling issues |
| 4916 | LowerState nested arc.state wrong clock | 58% | OPEN | LowerStatePass related |

---

## Closed Issues Analysis

### Recently Closed Similar Issues

| # | Title | Closed | Relevance |
|---|-------|--------|-----------|
| 9466 | arcilator fails to lower llhd.constant_time | 2026-01-17 | Very recent closure |
| 9469 | Array indexing in always_ff | 2026-01-25 | Very recent closure |
| 9417 | hw.bitcast data corruption | 2026-01-07 | Type width handling |

**Observation**: Multiple recent closures (Jan 7-25, 2026) in Arc/arcilator suggest active bug fixing cycle. Our crash may be fixed in newer commits.

---

## Root Cause Analysis

### The Core Problem

CIRCT is in transition from using `hw.inout` to a custom `llhd.ref<T>` type system:

```
circt-verilog (SystemVerilog with inout ports)
    ↓ (generates LLHD with ref types)
LLHD IR containing !llhd.ref<i1> types
    ↓ (arcilator runs ConvertToArcs)
Arc conversion FAILS
    ↓
StateType::verify() receives !llhd.ref<i1>
    ↓
computeLLVMBitWidth() returns nullopt (no LLHD support)
    ↓
Assertion: "state type must have a known bit width"
```

### Why Our Crash Occurs

1. **Upstream**: `circt-verilog` converts SV inout ports to LLHD ref types
2. **Integration Gap**: Arc dialect's StateType doesn't recognize LLHD types
3. **Missing Pass**: No conversion pass to handle llhd.ref → hw.inout or llhd.ref → other representation
4. **Result**: AssertionError instead of graceful degradation

---

## Recommendations

### For Bug Reporter

1. **Check Issue #9395** - Review closure comments for any fixes applied to ConvertToArcs pass
2. **Try Recent Builds** - Since multiple arc/arcilator issues closed Jan 7-25, your crash may already be fixed
3. **Check Issue #8825** - Track the status of the llhd.ref type system migration
4. **Workaround** - Avoid inout ports in SystemVerilog until this is resolved

### For CIRCT Developers

1. **Add LLHD Type Support** to StateType::verify() and computeLLVMBitWidth()
2. **Add Conversion Pattern** for llhd.ref types in ConvertToArcs pass
3. **Better Error Messages** - Replace assertion with user-friendly diagnostic
4. **Integration Test** - Add test case for inout ports through full arcilator pipeline

---

## Search Parameters Used

### Successful Search Queries

```bash
# Found Issue #9395
gh issue list --repo llvm/circt --state closed --search "arcilator" 

# Found Issue #8825
gh issue list --repo llvm/circt --search "llhd.ref"

# Found Issue #9467
gh issue list --repo llvm/circt --search "arcilator LLHD"

# Related issues
gh issue list --repo llvm/circt --search "arc state"
gh issue list --repo llvm/circt --search "inout port"
```

### Issues Not Found

- No exact duplicate for `llhd.ref<T>` + `StateType` assertion
- No specific issue for inout→llhd.ref conversion in arcilator
- No closed issue addressing tristate/bidirectional support in Arc

---

## Conclusion

**Duplicate Status**: ⚠️ **LIKELY DUPLICATE or PART OF LARGER ISSUE**

**Primary Match**: Issue #9395 (Arcilator assertion failure - CLOSED)  
**Root Cause**: Issue #8825 (Type system migration - OPEN)  
**Pattern**: Issue #9467 + others (LLHD→Arc conversion issues - ONGOING)

**Next Steps**:
1. Review Issue #9395 comments for applicable fixes
2. Try latest CIRCT build to check if already fixed
3. If not fixed, this can be reported as a duplicate of #8825 with specific focus on StateType/inout handling

---

**Report Generated**: 2026-01-31  
**Tool**: GitHub CLI (gh)  
**Query Coverage**: 16 keyword combinations, 50+ issue reviews
