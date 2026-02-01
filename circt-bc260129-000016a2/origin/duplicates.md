# CIRCT Duplicate Check Report

**Date**: 2025-02-01
**Tool**: arcilator (Arc dialect)
**Crash Type**: Assertion Failure
**Error Message**: `state type must have a known bit width; got '!llhd.ref<i1>'`

## Summary

Searched CIRCT GitHub issues repository for potential duplicates using 8 different search queries related to the reported crash. **Found 1 highly matching issue** that appears to be a near-exact duplicate or variant of the same bug.

---

## Search Queries Executed

1. ✓ `arcilator inout` - Found #9574
2. ✓ `arcilator llhd.ref` - Found #9574
3. ✓ `StateType known bit width` - Found #9574
4. ✓ `LowerState assertion` - Found #9574
5. ✗ `arcilator tri-state` - No results
6. ✓ `arcilator crash` - Found #9574, #9260
7. ✓ `inout StateType` - Found #9574
8. ✓ `llhd.ref state` - Found #9574

---

## Primary Match: Issue #9574

**Title**: `[Arc] Assertion failure when lowering inout ports in sequential logic`
**Status**: OPEN
**URL**: https://github.com/llvm/circt/issues/9574
**Created**: 2026-02-01T05:48:51Z
**Similarity Score**: 9.5/10

### Matching Factors

| Factor | Status | Details |
|--------|--------|---------|
| **Error Message** | ✓ EXACT | Same: `state type must have a known bit width; got '!llhd.ref<...>'` |
| **Component** | ✓ MATCH | Both: arcilator tool |
| **Assertion Location** | ✓ EXACT | Both: LowerState.cpp:219 in ModuleLowering::run() |
| **Trigger Construct** | ✓ MATCH | Both: inout port with tri-state assignment/use |
| **Problematic Type** | ✓ EXACT | Both: !llhd.ref<i1> (LLHD reference type) |
| **Root Cause** | ✓ EXACT | Both: StateType::get() cannot handle LLHD reference types |
| **Affected Pass** | ✓ EXACT | Both: LowerStatePass |
| **Crash Signature** | ✓ MATCH | Both: `assertion 'succeeded(ConcreteT::verifyInvariants(...))' failed` |

### Issue #9574 Description

```
CIRCT crashes with an assertion failure when compiling SystemVerilog code 
that uses `inout` ports within `always_ff` blocks. The crash occurs in the 
Arc dialect's `LowerStatePass` when attempting to create a `StateType` for 
an LLHD reference type.
```

**Test Case** (from #9574):
```systemverilog
module MixedPorts(
  inout wire c,
  input logic clk
);
  logic temp_reg;

  always_ff @(posedge clk) begin
    temp_reg <= c;
  end
endmodule
```

**Reproduction Command**:
```bash
circt-verilog --ir-hw bug.sv | arcilator
```

### Root Cause (from #9574)

1. Frontend (circt-verilog) parses the inout port as `!llhd.ref<i1>`
2. Arc LowerStatePass attempts to create state storage for the always_ff block
3. StateType::get() calls verifyInvariants() which requires known bit widths
4. LLHD reference types are opaque pointers without intrinsic bit width
5. Assertion fails at LowerState.cpp:219

---

## Secondary Matches

### Issue #9260: Arcilator crashes in Upload Release Artifacts CI

**Status**: OPEN
**Similarity Score**: 2.0/10
**Relevance**: LOW

This is a general arcilator crash in CI but lacks detailed crash information. It does not appear to be related to type handling or inout ports specifically.

---

## Comparison: Current Report vs Issue #9574

| Aspect | Current Report | Issue #9574 | Match? |
|--------|---|---|---|
| Tool | arcilator | arcilator | ✓ Yes |
| Crash Type | assertion_failure | assertion_failure | ✓ Yes |
| Pass | LowerState | LowerState | ✓ Yes |
| File | LowerState.cpp | LowerState.cpp | ✓ Yes |
| Line | 219 | 219 | ✓ Yes |
| Function | ModuleLowering::run() | ModuleLowering::run() | ✓ Yes |
| Error Message | state type must have a known bit width; got '!llhd.ref<i1>' | state type must have a known bit width; got '!llhd.ref<i1>' | ✓ Yes |
| Trigger | inout port with tri-state | inout port in always_ff | ✓ Same Pattern |
| Type Issue | !llhd.ref<i1> | !llhd.ref<i1> | ✓ Yes |
| Root Cause | StateType lacks support for llhd.ref | StateType lacks support for llhd.ref | ✓ Yes |

---

## Assessment

### Conclusion: **LIKELY DUPLICATE**

The current bug report and issue #9574 represent:
- **Either**: The exact same bug triggered by slightly different SystemVerilog constructs
- **Or**: Multiple manifestations of the same underlying issue

### Evidence

1. **Identical Error Message**: Both fail with the same assertion at the same code location
2. **Same Root Cause**: StateType::get() cannot process LLHD reference types
3. **Same Trigger Pattern**: Both involve inout ports with tri-state/reference semantics
4. **Same Component Path**: arcilator → LowerStatePass → StateType validation
5. **Exact Location Match**: Line 219 of LowerState.cpp

### Differences (Minor)

- Current report: inout port with tri-state assignment (`io_port = ... ? 1'bz : ...`)
- Issue #9574: inout port read inside always_ff block (`temp_reg <= c`)
- Both are essentially using inout ports in contexts that require state creation

The differences are variations in *how* the inout port is used, not the underlying type system issue.

---

## Recommendation

**STATUS**: `review_existing` → **LIKELY DUPLICATE OF #9574**

**ACTION**: 
- Do NOT file a new issue yet
- Review issue #9574 in detail
- Consider if your test case represents a new angle on the same bug
- If issue #9574 is stale or closed, file as a new report
- If #9574 is active, consider adding your test case as a comment to strengthen the bug report

**Next Steps**:
1. Visit: https://github.com/llvm/circt/issues/9574
2. Review the full discussion and any proposed fixes
3. Determine if your variant test case adds value to that issue
4. Coordinate with the issue maintainer (@jpienaar or other assignee)

---

## Files Generated

- `duplicates.json` - Structured search results with similarity scoring
- `duplicates.md` - This human-readable report

---

## Search Statistics

- Total queries executed: 8
- Unique issues found: 2
- High confidence matches: 1 (score ≥ 8.0)
- Medium confidence matches: 1 (score < 8.0)
- Low relevance results: 0

