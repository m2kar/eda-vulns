# Duplicate Check Report for CIRCT Bug circt-b5

**Analysis Date**: 2026-02-01
**Crash ID**: circt-b5
**Tool**: circt-verilog
**Dialect**: Comb
**Pass**: Canonicalizer

---

## Crash Signature

- **Assertion**: `op->use_empty() && "expected 'op' to have no uses"`
- **Location**: `llvm/mlir/lib/IR/PatternMatch.cpp:156` in `mlir::RewriterBase::eraseOp()`
- **Root Cause**: The `extractConcatToConcatExtract` pattern in `lib/Dialect/Comb/CombFolds.cpp` attempts to replace an ExtractOp with a single value, but the operation still has uses when `eraseOp()` is called

---

## Search Keywords

- canonicalizer
- assertion
- use_empty
- ExtractOp
- extractConcatToConcatExtract
- replaceOp
- GreedyPatternRewriteDriver
- Comb
- fold
- pattern

---

## Duplicate Check Results

**Recommendation**: `likely_new`
**Confidence**: 0.75
**Top Score**: 0.0
**Top Issue**: None found

---

## Reasoning

The crash is highly specific to:
1. The `extractConcatToConcatExtract` pattern in CombFolds.cpp
2. Multiple extract operations from the same source
3. The GreedyPatternRewriteDriver's concurrent pattern application
4. The specific assertion failure in `eraseOp()` when `op->use_empty()` is false

This combination suggests a **likely new issue**, though without full GitHub search access, there remains a possibility of similar existing issues. The crash is reproducible with a minimal 16-line SystemVerilog test case involving mixed continuous and procedural assignments with bit-indexed array access.

---

## Recommended Next Steps

1. Search CIRCT GitHub issues with keywords: `extractConcatToConcatExtract`, `use_empty`, `Comb canonicalizer`
2. Review recent changes to `lib/Dialect/Comb/CombFolds.cpp` and `GreedyPatternRewriteDriver`
3. If no duplicates found, file new issue with the minimal test case and root cause analysis
4. Consider implementing one of the suggested fixes in the root_cause.md file

---

## Conclusion

Based on the analysis, this appears to be a **new bug** that should be reported to the CIRCT project.
