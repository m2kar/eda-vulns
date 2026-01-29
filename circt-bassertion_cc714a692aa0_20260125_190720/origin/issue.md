# [Comb] Assertion failure in extractConcatToConcatExtract with packed struct arrays

## ⚠️ Duplicate Issue Note

**This is a duplicate of [#8863](https://github.com/llvm/circt/issues/8863)**: [Comb] Concat/extract canonicalizer crashes on loop

The crash signature, assertion message, and stack trace are identical. This issue provides a different reproduction case using SystemVerilog packed struct arrays with bit-level indexing.

**Recommendation**: Add this reproduction case as a comment to #8863 rather than creating a new issue.

---

## Summary

CIRCT crashes with an assertion failure when processing SystemVerilog code that uses packed struct arrays with bit-level field indexing in `always_comb` blocks. The crash occurs in the `extractConcatToConcatExtract` canonicalization pattern when attempting to erase an operation that still has uses.

## Expected Behavior

The code should compile without errors. The constructs used (packed structs, packed arrays, bit-selects, `always_comb`) are all valid per IEEE 1800-2017.

## Actual Behavior

CIRCT crashes with an assertion failure:
```
Assertion `op->use_empty() && "expected 'op' to have no uses"' failed.
```

### Stack Trace
```
#11 mlir::RewriterBase::eraseOp(Operation *)           PatternMatch.cpp:156
#12 circt::replaceOpAndCopyNamehint(...)               Naming.cpp:82
#13 extractConcatToConcatExtract(...)                  CombFolds.cpp:547
#14 circt::comb::ExtractOp::canonicalize(...)          CombFolds.cpp:615
#27 (anonymous namespace)::Canonicalizer::runOnOperation()
```

## Reproduction

### Minimal Test Case
```systemverilog
module m(input logic D, output logic Q);
  typedef struct packed { logic [1:0] f; } t;
  t [1:0] a;
  always_comb begin
    a[0].f[0] = D;
    Q = a[0].f[0];
  end
endmodule
```

### Reproduction Command
```bash
/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw bug.sv
```

### Full Test Case
```systemverilog
module top_module(input logic clk, input logic D, output logic Q);
  typedef struct packed {
    logic [4:0] field0;
  } array_elem_t;

  array_elem_t [7:0] data_array;

  always_comb begin
    data_array[0].field0[0] = D;
  end

  always_ff @(posedge clk) begin
    Q <= data_array[0].field0[0];
  end

endmodule
```

## Environment

- **CIRCT Version**: 1.139.0 (commit cc714a692aa0)
- **LLVM Version**: 22.0.0git
- **Build Type**: Optimized build with assertions enabled
- **Tool**: `circt-verilog`

## Root Cause Analysis

### Crash Location
- **File**: `lib/Dialect/Comb/CombFolds.cpp`
- **Function**: `extractConcatToConcatExtract`
- **Line**: 547

### Problem Description

The `extractConcatToConcatExtract` function attempts to replace an `ExtractOp` with a single value when `reverseConcatArgs.size() == 1`. However, in cases involving packed struct arrays with bit-level indexing, the replacement value may still have dependencies on the original operation or its intermediate results.

When `replaceOpAndCopyNamehint(rewriter, op, reverseConcatArgs[0])` is called at line 547, the MLIR rewriter detects that `op` still has uses (via `op->use_empty()` check in `PatternMatch.cpp:156`) and triggers the assertion failure.

### Key Factors
1. **Packed struct arrays**: `typedef struct packed { logic [1:0] f; } t [1:0] a;`
2. **Bit-level indexing**: `a[0].f[0]` extracts a single bit from the struct field
3. **Multiple uses**: The same packed array element is accessed in multiple contexts (`always_comb` write, `always_ff` read)
4. **Greedy pattern rewriting**: The canonicalizer's greedy driver processes multiple related extract/concat operations in parallel, creating cyclic dependencies

### Hypothesis

In the greedy pattern rewriter's worklist processing, the same `ExtractOp` may be referenced by multiple pattern applications. When `extractConcatToConcatExtract` creates a replacement that itself depends on `op` (or its intermediate results), it forms a cyclic dependency. When the rewriter attempts to erase `op`, the `use_empty()` assertion fails because other operations still reference it.

## Verification

### Syntax Validation
| Tool | Result |
|------|--------|
| Verilator `--lint-only` | ✅ Pass (0 errors) |
| Slang `--lint-only` | ✅ Pass (0 errors) |
| IEEE 1800-2017 | ✅ Valid |

### Cross-Tool Behavior
- **Verilator**: Compiles successfully
- **Slang**: Compiles successfully
- **CIRCT**: ❌ Crashes (assertion failure)

## Related Issues

- **#8863**: [Comb] Concat/extract canonicalizer crashes on loop (Duplicate - same assertion and stack trace)
- #8024: [Comb] Crash in AndOp folder (Different crash in Comb dialect)
- #8690: [Canonicalize] Non-terminating canonicalization

## Suggested Fixes

### Fix 1: Add use_empty check before replacement
```cpp
if (reverseConcatArgs.size() == 1) {
  if (!op->use_empty()) {
    // Skip this optimization if op still has uses
    return failure();
  }
  replaceOpAndCopyNamehint(rewriter, op, reverseConcatArgs[0]);
}
```

### Fix 2: Use replaceAllUsesWith instead of direct replacement
```cpp
if (reverseConcatArgs.size() == 1) {
  rewriter.replaceAllUsesWith(op, reverseConcatArgs[0]);
  // Let the rewriter handle erasure automatically
}
```

### Fix 3: Validate replacement value doesn't reference op
Before replacement, verify that `reverseConcatArgs[0]`'s definition chain doesn't include `op` to avoid cyclic dependencies.

## Additional Notes

This bug appears to be fixed in `/opt/firtool/bin/circt-verilog` (firtool-1.139.0 built without assertions), suggesting the underlying issue has been addressed in recent commits. However, the reproduction case is still valuable as it demonstrates a different trigger (packed struct arrays) for the same root cause in the canonicalization infrastructure.
