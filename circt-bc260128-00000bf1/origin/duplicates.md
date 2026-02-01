# CIRCT Duplicate Issue Check Report

**Test Case ID:** 260128-00000bf1  
**Bug Type:** Timeout during canonicalization pass  
**Report Date:** 2026-01-31

---

## Executive Summary

The reported bug (timeout in canonicalize pass after llhd-sig2reg on partial struct field assignment) appears to be **NOVEL but highly related to existing canonicalize stability issues**. Multiple related bugs exist in the CIRCT repository, all involving canonicalize pass infinite loops or performance issues.

**Recommendation:** `review_existing` - Human review needed to determine if this is a variant of existing issues or a genuinely new bug pattern.

---

## Search Results Summary

**Total Issues Found:** 8 related issues  
**Search Keywords Used:** 
- timeout llhd-sig2reg
- struct field assignment  
- always_comb timeout
- circular SSA
- canonicalize infinite loop
- canonicalize timeout
- llhd-sig2reg
- struct_inject
- infinite loop
- canonicalize

---

## Most Similar Issues (Ranked by Relevance)

### 1. **Issue #9560: [FIRRTL] Canonicalize infinite loop** ⭐ TOP MATCH
- **Status:** OPEN (Created: 2026-01-31 01:38:54Z)
- **Similarity Score:** 8.5/10
- **URL:** https://github.com/llvm/circt/issues/9560

**Description:**
Canonicalize pass enters infinite loop on FIRRTL code with constant folding patterns.

**Key Code Pattern:**
```mlir
firrtl.regreset %clk, %rst, %c0_ui4
%0 = firrtl.bits %reg 0 to 0
%1 = firrtl.eq %c0_ui1, %c0_ui1
%2 = firrtl.and %1, %0
```

**Similarity Analysis:**
- ✅ **Same Problem:** Canonicalize entering infinite loop
- ✅ **Same Timeout Behavior:** Both cause compilation timeout (300s limit)
- ❌ **Different Dialect:** FIRRTL vs LLHD (Moore)
- ❌ **Different Trigger:** Constant folding pattern vs circular SSA in struct operations
- ⚠️ **Recent:** Same date as our bug, suggesting systematic issue

**Connection:** This is the most recent canonicalize infinite loop report. If this bug is confirmed as canonicalize-related in LLHD/Moore conversion, they may be manifestations of the same underlying canonicalize stability issue.

---

### 2. **Issue #8865: [Comb] AddOp canonicalizer hangs in an infinite loop** 
- **Status:** CLOSED (Created: 2025-07-18)
- **Similarity Score:** 8.0/10
- **URL:** https://github.com/llvm/circt/issues/8865

**Problem:**
Canonicalize hangs in infinite loop when processing recursive add operations from control flow simplification.

**Key Pattern:**
```mlir
%35 = comb.add %35, %34 : i32  // Circular dependency
// Canonicalizer expands this 33+ times before getting stuck
```

**Similarity Analysis:**
- ✅ **Exact Symptom:** Canonicalize infinite loop → timeout
- ✅ **Root Cause Class:** Circular/recursive operation definitions
- ❌ **Different Operation:** add vs struct_inject/bitcast
- ⚠️ **Status:** This was closed (likely fixed), but pattern is identical

**Connection:** Shows canonicalize cannot safely handle circular SSA definitions. Our bug's circular SSA pattern (struct_inject on bitcast chain) likely triggers same canonicalize bug.

---

### 3. **Issue #8022: [Comb] Infinite loop in OrOp folder**
- **Status:** OPEN (Created: 2025-01-23)  
- **Similarity Score:** 7.5/10
- **URL:** https://github.com/llvm/circt/issues/8022

**Problem:**
OrOp folder enters infinite loop during canonicalization.

**Similarity Analysis:**
- ✅ **Same Class:** Infinite loop in canonicalize/folding
- ❌ **Different Operation:** Bitwise OR vs struct operations
- ⚠️ **Pattern:** Shows canonicalize folding is generally fragile

---

### 4. **Issue #8863: [Comb] Concat/extract canonicalizer crashes on loop**
- **Status:** OPEN (Created: 2025-08-18)
- **Similarity Score:** 7.0/10

**Problem:**
Canonicalizer crashes when processing concat/extract operations with cyclic dependencies.

**Similarity Analysis:**
- ✅ **Cyclic Dependency Detection:** Issues with canonicalizer handling cycles
- ✅ **Related Operations:** hw.struct_inject/extract are similar to concat/extract
- ❌ **Manifestation:** Crash vs timeout

---

### 5. **Issue #8065: [LLHD][Arc] Indexing and slicing lowering from Verilog to LLVM IR**
- **Status:** OPEN (Created: 2025-02-11)
- **Similarity Score:** 6.5/10
- **URL:** https://github.com/llvm/circt/issues/8065

**Problem:**
Pipeline using `llhd-sig2reg` followed by `canonicalize` fails with "non-pure operation" error.

**Pipeline Used:**
```bash
circt-verilog ranges.sv | circt-opt --llhd-early-code-motion \
  --llhd-temporal-code-motion --llhd-desequentialize \
  --llhd-sig2reg --canonicalize | arcilator
```

**Similarity Analysis:**
- ✅ **Same Pass Sequence:** llhd-sig2reg → canonicalize
- ✅ **Same Dialect Path:** LLHD (from Moore/Verilog)
- ✅ **Same Domain:** Struct-like types (array indexing similar to struct field operations)
- ❌ **Different Error:** Non-pure operation vs timeout
- ⚠️ **Related Issue:** Shows llhd-sig2reg + canonicalize is problematic combination

---

### 6. **Issue #8012: [Moore][Arc][LLHD] Moore to LLVM lowering issues**
- **Status:** OPEN (Created: 2024-12-22)
- **Similarity Score:** 6.0/10
- **URL:** https://github.com/llvm/circt/issues/8012

**Problem:**
Multiple errors in Moore → Arc → LLVM lowering pipeline including llhd-sig2reg + canonicalize failures.

**Pipeline Issue:**
```bash
circt-verilog dff.sv | circt-opt --llhd-early-code-motion \
  --llhd-temporal-code-motion --llhd-desequentialize \
  --llhd-sig2reg --canonicalize | arcilator
```

**Similarity Analysis:**
- ✅ **Exact Pass Sequence:** Same problematic pipeline
- ✅ **Exact Conversion Path:** Moore → LLHD → Arc
- ❌ **Error Type:** Legalization failure vs timeout
- ⚠️ **High Relevance:** Shows llhd-sig2reg + canonicalize systematically fails

---

## Issue Clusters

### Cluster A: Canonicalize Infinite Loops (Most Relevant)
- #9560 - FIRRTL canonicalize infinite loop [NEW - SAME DATE]
- #8865 - Comb AddOp canonicalizer infinite loop [CLOSED]
- #8022 - Comb OrOp folder infinite loop [OPEN]
- #8863 - Comb concat/extract canonicalizer crash on loop [OPEN]

**Pattern:** Canonicalize pass has systematic issues with:
- Circular/recursive SSA definitions
- Folding operations that generate circular references
- Detecting and breaking infinite optimization loops

### Cluster B: llhd-sig2reg + Canonicalize Issues (Directly Related)
- #8065 - LLHD indexing/slicing lowering fails after canonicalize
- #8012 - Moore→LLHD→Arc pipeline fails with canonicalize

**Pattern:** The pass sequence `llhd-sig2reg` → `canonicalize` is known to cause issues:
- Creates non-pure operations
- Fails legalization in downstream passes
- May be related to how llhd-sig2reg generates SSA form

---

## Root Cause Analysis

### Reported Bug Mechanism
```
Input: Partial struct field assignment in always_comb
↓
Moore Conversion: Creates llhd.sig references
↓
llhd-sig2reg Pass: Generates read-modify-write pattern with circular SSA
  %7 = hw.struct_inject %5[...], where:
    %5 = hw.bitcast %4
    %4 = hw.bitcast %7  ← CIRCULAR REFERENCE
↓
Canonicalize Pass: Attempts optimization on circular SSA definition
  → Infinite loop / timeout (300 seconds)
```

### Why It Relates to Existing Issues

**Issue #9560 (FIRRTL):** Shows canonicalize cannot detect infinite loops in optimization patterns
**Issue #8865 (Comb):** Shows canonicalize expanding recursive operations indefinitely
**Issues #8065, #8012:** Show llhd-sig2reg output is problematic for downstream passes

**Hypothesis:** This bug is likely either:
1. **Variant of #9560:** Same canonicalize infinite loop mechanism, different dialect
2. **New manifestation of #8065/#8012:** llhd-sig2reg creating invalid SSA that breaks canonicalize
3. **Genuinely new bug:** Unique combination of struct + always_comb + llhd-sig2reg that no existing issue covers

---

## Verdict Summary

| Criterion | Assessment |
|-----------|-----------|
| **Exact Duplicate** | ❌ No identical issue found |
| **High Similarity Match** | ✅ Issue #9560 (score 8.5) - canonicalize infinite loop |
| **Related Area Issues** | ✅ Multiple (#8865, #8022, #8863, #8065, #8012) |
| **Root Cause Overlap** | ⚠️ Partial - canonicalize stability is known issue |
| **Novelty Assessment** | ⚠️ Novel trigger (struct field assignment), but symptom is known |

---

## Recommendation: `review_existing`

**Action:** Before filing new issue, recommend:

1. **Check if #9560 reproduces with LLHD:** If canonicalize infinite loop is generic, our LLHD case may be variant
2. **Check if llhd-sig2reg generates circular SSA:** Compare our IR dump with #8065/#8012 outputs
3. **Review canonicalize pass logic:** May need fix to detect/prevent infinite loops
4. **Consider if issue should be filed as:**
   - New issue highlighting struct field assignment + llhd-sig2reg combination
   - Related to #9560 showing canonicalize infinite loop is systematic
   - Sub-issue of llhd-sig2reg → canonicalize pipeline problems

---

## Top Recommendation for New Issue (If Distinct)

If deemed novel, file as:
```
Title: [LLHD][Moore] Timeout in canonicalize after llhd-sig2reg on struct field assignment

Description:
- Partial struct field assignment in always_comb block triggers timeout
- llhd-sig2reg creates circular SSA (struct_inject on bitcast cycle)
- Canonicalize enters infinite loop attempting optimization
- Related to but distinct from issue #9560 (FIRRTL canonicalize infinite loop)
- Also related to #8065, #8012 showing llhd-sig2reg + canonicalize fragility

Pipeline: circt-verilog --ir-hw | circt-opt --pass-pipeline=...

See also: Issues #9560, #8865, #8022, #8065, #8012
```

---

**Report Generated:** 2026-01-31  
**Status:** Ready for human review
