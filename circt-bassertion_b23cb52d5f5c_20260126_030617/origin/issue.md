# [Comb] Assertion failure in extractConcatToConcatExtract: expected 'op' to have no uses

## ⚠️ DUPLICATE ISSUE - DO NOT SUBMIT

**This bug has already been reported as Issue #8863**: "[Comb] Concat/extract canonicalizer crashes on loop"

- URL: https://github.com/llvm/circt/issues/8863
- Similarity Score: **15.5** (Exact match on assertion and stack trace)

---

## Crash Type
`assertion`

## CIRCT Version
CIRCT firtool-1.139.0

## Crash Hash
`b23cb52d5f5c`

## Affected Dialect
Comb

## Error Message

```
Assertion `op->use_empty() && "expected 'op' to have no uses"` failed
```

## Stack Trace

```
circt-verilog: /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/lib/IR/PatternMatch.cpp:156: virtual void mlir::RewriterBase::eraseOp(Operation *): Assertion `op->use_empty() && "expected 'op' to have no uses"' failed.

Stack dump:
 #12 circt::replaceOpAndCopyNamehint(mlir::PatternRewriter&, mlir::Operation*, mlir::Value) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Support/Naming.cpp:82:1
 #13 extractConcatToConcatExtract(circt::comb::ExtractOp, circt::comb::ConcatOp, mlir::PatternRewriter&) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Dialect/Comb/CombFolds.cpp
 #14 circt::comb::ExtractOp::canonicalize(circt::comb::ExtractOp, mlir::PatternRewriter&) /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/lib/Dialect/Comb/CombFolds.cpp:615:12
 #27 mlir::PatternApplicator::matchAndRewrite
 #40 (anonymous namespace)::Canonicalizer::runOnOperation()
```

## Reproduction Command

```bash
circt-verilog --ir-hw source.sv
```

**Note**: Current `source.sv` may be over-simplified. The original fuzzer test case triggered the concat/extract optimization path, but the simplified version may have lost the triggering pattern.

## Test Case

```systemverilog
module test_module(input logic in);
  logic [7:0] s;
  logic x;

  always_comb begin
    x = in;
    s[0] = x;
  end
endmodule
```

## Root Cause Analysis

### Problem
CIRCT crashes during comb dialect canonicalization when processing Extract/Concat operations. The assertion failure indicates an operation is being erased while still having active uses, violating MLIR pattern rewriter invariants.

### Crashing Function
`extractConcatToConcatExtract` in `lib/Dialect/Comb/CombFolds.cpp:546-547`

### Mechanism
When the optimization detects that `reverseConcatArgs.size() == 1`, it directly replaces the `ExtractOp` with a single operand using `replaceOpAndCopyNamehint()`. However, this operand may still have active uses elsewhere in the IR, causing the assertion in `RewriterBase::eraseOp()` to fail.

### Root Cause Hypothesis (High Confidence)
The `extractConcatToConcatExtract` optimization reuses a ConcatOp operand as the replacement value without verifying that the operation being replaced has no conflicting uses. When `rewriter.replaceOp()` attempts to erase the original operation, the MLIR invariant that `op->use_empty()` must be true is violated.

## Validation Results

| Check | Result |
|--------|--------|
| Syntax Check (Slang) | ✓ PASS |
| Cross-Tool Validation | ✓ PASS (Verilator, Icarus) |
| Classification | `valid_testcase` + `genuine_bug` |

The test case is syntactically valid SystemVerilog accepted by all major tools. The crash is a genuine CIRCT compiler bug in the pattern rewriter.

## Related Files

- `lib/Dialect/Comb/CombFolds.cpp` - Contains `extractConcatToConcatExtract`
- `lib/Support/Naming.cpp` - Contains `replaceOpAndCopyNamehint`
- `llvm/mlir/lib/IR/PatternMatch.cpp` - Contains failing `eraseOp` assertion

## Suggested Fix

1. Add use-chain validation before calling `replaceOpAndCopyNamehint` in `extractConcatToConcatExtract`
2. Ensure the operation has no uses before attempting to erase it
3. Add defensive checks at `CombFolds.cpp:546-547` before replacement

## Keywords

`concat` `extract` `canonicalization` `eraseOp` `use-chain` `CombFolds` `PatternMatch`

## Duplicate Status

**This is a duplicate of Issue #8863**

Similarity analysis shows:
- Exact assertion match: `expected 'op' to have no uses`
- Same stack trace: `extractConcatToConcatExtract` → `replaceOpAndCopyNamehint` → `eraseOp`
- Same crash location: `CombFolds.cpp`
- Same dialect: Comb

**Recommendation**: Monitor Issue #8863 for fix progress. Do not submit this as a new issue.
