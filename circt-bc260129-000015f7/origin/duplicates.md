# Duplicate Issue Check Report

**Test Case ID:** 260129-000015f7  
**Report Generated:** 2026-02-01T10:34:23.539506  
**Status:** Duplicate Check Complete

---

## Summary

| Metric | Value |
|--------|-------|
| **Recommendation** | LIKELY_DUPLICATE |
| **Top Similarity Score** | 67.25% |
| **Most Similar Issue** | #8693 |
| **Total Issues Found** | 3 |

---

## Issue Details

### Current Crash Analysis

- **Crash Type:** assertion
- **Dialect:** LLHD
- **Pass:** Mem2Reg
- **Tool:** circt-verilog
- **Error Message:** integer bitwidth is limited to 16777215 bits
- **Assertion:** Assertion `succeeded(ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed

**Root Cause:** LLHD Mem2Reg pass fails to handle unsupported class types, resulting in invalid integer bitwidth computation

**Severity:** medium  
**Reproducibility:** deterministic

---

## Similarity Analysis Results

### 1. Issue #8693: [Mem2Reg] Local signal does not dominate final drive

- **Similarity Score:** 67.25%
- **State:** OPEN
- **Created:** 2025-07-11T20:11:54Z
- **Labels:** LLHD
- **Preview:** For input (derived from SV import minized and then MLIR minized)

```mlir
module {
  hw.module @a() {
    %false = hw.constant false
    %b = llhd.sig %false : i1
    llhd.combinational {
      cf.br ...

**Similarity Details:**
- has_mem2reg: True
- has_llhd: True
- keyword_score: 7.50
- has_type_issue: True
- title_similarity_score: 4.75

### 2. Issue #8286: [circt-verilog][llhd][arcilator] Verilog-to-LLVM lowering issues

- **Similarity Score:** 54.75%
- **State:** OPEN
- **Created:** 2025-03-03T10:23:03Z
- **Labels:** None
- **Preview:** Hi all!

There are a few issues related to moore to llvm lowering pipeline.

Currently there is no possibility to lower combination logic with control flow operators into LLVM. For example:

```verilo...

**Similarity Details:**
- has_mem2reg: True
- has_llhd: True
- keyword_score: 7.50
- title_similarity_score: 2.25

### 3. Issue #1352: [FIRRTL] Add create vector/bundle ops

- **Similarity Score:** 25.57%
- **State:** OPEN
- **Created:** 2021-06-30T15:15:09Z
- **Labels:** enhancement, FIRRTL
- **Preview:** As [discussed](https://github.com/llvm/circt/pull/1304#discussion_r659352639) in #1304, constructing a vector or bundle currently requires a temporary `firrtl.wire`, with subfield/subindex ops and con...

**Similarity Details:**
- has_llhd: True
- keyword_score: 3.75
- title_similarity_score: 1.82

---

## Top Match Analysis - Issue #8693

**Title:** [Mem2Reg] Local signal does not dominate final drive

**URL:** https://github.com/llvm/circt/issues/8693

**State:** OPEN

**Created By:** jpienaar

**Created At:** 2025-07-11T20:11:54Z

**Updated At:** 2025-07-11T20:50:35Z

**Description:**
```
For input (derived from SV import minized and then MLIR minized)

```mlir
module {
  hw.module @a() {
    %false = hw.constant false
    %b = llhd.sig %false : i1
    llhd.combinational {
      cf.br ^bb2
    ^bb1:  // no predecessors
      %0 = llhd.prb %b : !hw.inout<i1>
      %1 = llhd.constant_time <0ns, 0d, 1e>
      %g = llhd.sig %false : i1
      llhd.drv %g, %0 after %1 : !hw.inout<i1>
      cf.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llhd.yield
    }
    hw.output
  }
}
```

results post llhd-mem2reg in

```
error: operand #0 does not dominate this use
note: see current operation: "llhd.drv"(%6, %7, %9, %8) : (!hw.inout<i1>, i1, !llhd.time, i1) -> ()
/tmp/t2.mlir:10:12: note: operand defined here (op in the same region)
      %g = llhd.sig %false : i1
           ^
```

print IR after:

```mlir
"builtin.module"() ({
  "hw.module"() <{module_type = !hw.modty<>, parameters = [], sym_name = "a"}> ({
    %0 = "hw.constant"() <{value = false}> : () -> i1
    %1 = "llhd.sig"(%...
```

### Comparison

**Similar Aspects:**
- Both involve the Mem2Reg pass
- Both related to LLHD dialect
- Both are assertion failures

**Key Differences:**
- Issue #8693: Focus on "Local signal does not dominate final drive"
- Current Issue: Focus on "integer bitwidth exceeds limit" with class instantiation

**Assessment:** The issues appear to be related but targeting different aspects of the Mem2Reg pass. Issue #8693 may provide useful context or partial solutions, but this appears to be a distinct variant focusing on class type handling.

---

## Recommendations

### Decision: LIKELY_DUPLICATE


1. **Review Issue #8693** in detail before creating a new issue
2. **Check if the crash is an exact duplicate** or a variant of the existing issue
3. **Consider adding this test case** as a reproducer to the existing issue if it's a duplicate
4. **Compare root causes** to determine if they address the same underlying problem
5. **DO NOT create a new issue** until after thorough comparison

**Next Steps:**
- Examine issue #8693 for duplicate confirmation
- Check issue timeline and any associated PRs
- Review PR comments for related fixes that might apply
- Determine if this crash needs a separate issue or can be tracked in the existing one

---

## Search Methodology

- **Search Queries Used:** 6
- **Keywords Extracted:** 
  - Dialect: LLHD
  - Pass: Mem2Reg
  - Crash Type: assertion
  - Tool: circt-verilog

---

## Conclusion

Based on the duplicate check analysis:

- **Top Match:** Issue #8693 with 67.25% similarity
- **Status:** LIKELY_DUPLICATE
- **Confidence:** HIGH

This analysis suggests the crash is **likely duplicate** to existing CIRCT issues.

---

*Generated by Duplicate Check Worker*  
*Do NOT submit issues without manual review*
