# [Comb] Assertion failure in extractConcatToConcatExtract during canonicalization with array element access

<!--
  CIRCT Bug Report
  Bug ID: 260128-00000d1b
  Generated: 2026-02-01
  
  NOTE: This is a REGRESSION TEST CASE for a bug that appears to be FIXED.
  The bug is NOT reproducible with current LLVM 22 tools.
  Related to existing issue #8863 with 88% similarity score.
-->

## ⚠️ Important Notes

> **This is a regression test case, NOT a new bug report.**
> 
> - **Original crash version**: circt-1.139.0
> - **Current status**: Bug appears FIXED in current CIRCT (LLVM 22.0.0git)
> - **Related issue**: #8863 (88% similarity score)
> - **Purpose**: Preserved as regression test to prevent reoccurrence

---

## Description

The Comb dialect's `extractConcatToConcatExtract` canonicalization pattern crashes with an assertion failure when processing SystemVerilog code that writes to and reads from the same array element, using the read value in a conditional expression.

The crash occurs because the pattern attempts to erase an `ExtractOp` that still has uses - the rewriter hasn't finished updating all references before the erasure is attempted.

**Crash Type**: Assertion failure  
**Dialect**: Comb  
**Failing Pass**: Canonicalizer  
**Function**: `extractConcatToConcatExtract`  
**Location**: `lib/Dialect/Comb/CombFolds.cpp:548`

---

## Steps to Reproduce

1. Save the test case below as `test.sv`
2. Run:
   ```bash
   circt-verilog --ir-hw test.sv
   ```

**Note**: This crash was observed in circt-1.139.0. Current CIRCT versions (LLVM 22) complete successfully.

---

## Test Case

```systemverilog
// Minimal test case for CIRCT crash in extractConcatToConcatExtract
// Bug: Assertion `op->use_empty() && "expected 'op' to have no uses"` failed
// Triggered by: Array element write followed by read in mux condition
// Original CIRCT version: circt-1.139.0
// Status: Fixed in later versions (LLVM 22 tools)
module example_module(input logic a, output logic [5:0] b);
  logic [7:0] arr;
  always_comb begin
    arr[0] = a;
  end
  assign b = arr[0] ? a : 6'b0;
endmodule
```

### Key Constructs

| Pattern | Code | Description |
|---------|------|-------------|
| Array element write | `arr[0] = a` | Creates concat operations to update single bit |
| Array element read | `arr[0]` | Creates extract operations |
| Conditional with extracted bit | `arr[0] ? a : 6'b0` | Mux using extracted bit as selector |

---

## Expected vs Actual Behavior

**Expected**: The compiler should successfully lower the SystemVerilog to HW dialect IR without crashing.

**Actual (circt-1.139.0)**: Assertion failure during canonicalization:
```
mlir::RewriterBase::eraseOp(Operation *): 
Assertion `op->use_empty() && "expected 'op' to have no uses"' failed.
```

**Current (LLVM 22)**: Completes successfully with exit code 0.

---

## Error Output (Original Crash)

```
Assertion failed:
  op->use_empty() && "expected 'op' to have no uses"
  at llvm/mlir/lib/IR/PatternMatch.cpp:156

Stack Trace Summary:
#0  llvm::sys::PrintStackTrace
#11 mlir::RewriterBase::eraseOp(Operation*)
#12 circt::replaceOpAndCopyNamehint (lib/Support/Naming.cpp:82)
#13 extractConcatToConcatExtract (lib/Dialect/Comb/CombFolds.cpp:548)
#14 circt::comb::ExtractOp::canonicalize (lib/Dialect/Comb/CombFolds.cpp:615)
#19 GreedyPatternRewriteDriver::processWorklist
#28 (anonymous namespace)::Canonicalizer::runOnOperation()
```

---

## Root Cause Analysis

### Primary Hypothesis (High Confidence)

The `extractConcatToConcatExtract` function creates new `ExtractOp` operations as part of the replacement, but the original `ExtractOp` still has uses that haven't been updated by the rewriter when `eraseOp` is called.

**Mechanism**:
1. Original IR: `%extract = extract(%concat, 0)`, with `%extract` used by operation X
2. Pattern creates new extract from concat operand
3. Pattern calls `replaceOpAndCopyNamehint(rewriter, op, newValue)`
4. If operation X hasn't been processed/updated yet, `op` still has uses
5. `rewriter.eraseOp(op)` asserts because `!op->use_empty()`

**Evidence**:
- Stack trace shows `eraseOp` called from `replaceOpAndCopyNamehint`
- Function creates new `ExtractOp` operations in `reverseConcatArgs`
- Assertion explicitly checks `op->use_empty()` before erase
- Pattern runs during greedy rewrite where multiple ops may reference same values

### Secondary Hypothesis (Medium Confidence)

Multiple `ExtractOps` referencing the same `ConcatOp` cause ordering issues in the canonicalizer worklist - canonicalizing one extract may create inconsistent state for others still pending in the worklist.

---

## Environment

| Component | Version |
|-----------|---------|
| **CIRCT Version (original crash)** | circt-1.139.0 |
| **CIRCT Version (current test)** | firtool-1.139.0 (LLVM 22.0.0git) |
| **OS** | Linux |
| **Architecture** | x86_64 |

---

## Cross-Tool Validation

The test case is valid SystemVerilog:

| Tool | Status | Notes |
|------|--------|-------|
| slang 9.1.0 | ✅ Pass | Build succeeded: 0 errors, 0 warnings |
| Verilator 5.022 | ⚠️ Warning | WIDTHEXPAND warning (lint only, not syntax error) |
| Icarus Verilog | ✅ Pass | Compiled successfully |

---

## Stack Trace

<details>
<summary>Click to expand full stack trace</summary>

```
#0  llvm::sys::PrintStackTrace
#11 mlir::RewriterBase::eraseOp(Operation*)
#12 circt::replaceOpAndCopyNamehint (lib/Support/Naming.cpp:82)
#13 extractConcatToConcatExtract (lib/Dialect/Comb/CombFolds.cpp:548)
#14 circt::comb::ExtractOp::canonicalize (lib/Dialect/Comb/CombFolds.cpp:615)
#15-18 mlir::PatternApplicator::matchAndRewrite
#19 GreedyPatternRewriteDriver::processWorklist
#20-27 mlir::PassManager infrastructure
#28 (anonymous namespace)::Canonicalizer::runOnOperation()
```

</details>

---

## Related Issues

### 🔗 #8863 - [Comb] Concat/extract canonicalizer crashes on loop

**Similarity Score: 88% (17.5 / 20.0)**

This issue is **highly related** to #8863 which reports the identical crash:

| Aspect | This Bug | Issue #8863 |
|--------|----------|-------------|
| Assertion | `op->use_empty()` | `op->use_empty()` |
| Function | `extractConcatToConcatExtract` | `extractConcatToConcatExtract` |
| File | `CombFolds.cpp` | `CombFolds.cpp` |
| Dialect | Comb | Comb |
| Trigger | Array element write + read | Array element write + read |
| Pass | Canonicalizer | Canonicalizer |

**Recommendation**: This test case may serve as an additional regression test for issue #8863.

### Other Related Issues

- #4688 - `comb::ExtractOp::canonicalize` is very slow (performance issue)
- #8260 - Comb concat/extract canonicalizers (enhancement request)

---

## Suggested Fix Directions

1. **Use `rewriter.replaceOp` directly**: Instead of `replaceOpAndCopyNamehint`, use the standard `rewriter.replaceOp` which handles use replacement and deferred erasure properly.

2. **Check for uses before replacement**: Add a guard to verify the operation can be safely replaced:
   ```cpp
   if (op->hasOneUse() || /* safe replacement conditions */) {
     replaceOpAndCopyNamehint(rewriter, op, replacement);
   }
   ```

3. **Review `replaceOpAndCopyNamehint` implementation**: Ensure it properly uses `replaceOp` semantics without explicit erasure.

---

## Files to Investigate

| File | Reason |
|------|--------|
| `lib/Dialect/Comb/CombFolds.cpp` | Contains failing `extractConcatToConcatExtract` function |
| `lib/Support/Naming.cpp` | Contains `replaceOpAndCopyNamehint` helper |
| `llvm/mlir/lib/IR/PatternMatch.cpp` | Contains the failing assertion |

---

## Classification

**Type**: Regression Test Case

This test case is preserved to ensure the bug does not reappear in future CIRCT versions. The fix appears to have been made between circt-1.139.0 and the current LLVM 22 build.

---

## Metadata

| Property | Value |
|----------|-------|
| Bug ID | 260128-00000d1b |
| Report Date | 2026-02-01 |
| Classification | regression_test |
| Validation | syntax valid, feature supported |
| Duplicate Status | Related to #8863 (88% similarity) |

---

*This issue report was generated with assistance from an automated CIRCT bug reporter. Test case preserved for regression testing purposes.*
