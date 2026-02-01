# Minimize Report

## Summary

| Metric | Value |
|--------|-------|
| Original File | source.sv (451 bytes) |
| Minimized File | bug.sv (200 bytes) |
| Reduction | 55.7% |
| Crash Type | timeout |
| Timeout Duration | 30 seconds |

## Minimization Process

### Initial Analysis

Based on analysis.json, the suspected triggers were:
1. Three-level nested module structure (NestedA → NestedB → NestedC)
2. Function call chain (func1 → func2)
3. always_comb with bit-level dependency (data[0] depends on data[7])

### Minimization Steps

| Step | Test | Result | Conclusion |
|------|------|--------|------------|
| 1 | Remove function chain (single func) | Timeout | Function chain not required |
| 2 | Remove function entirely | Timeout | Functions not required |
| 3 | Reduce to 2-level nesting | Timeout | 3-level nesting not required |
| 4 | Remove all nesting | Timeout | Nested modules not required |
| 5 | Test minimal case | Timeout | **Found minimal trigger** |

### Key Finding

**The original analysis was incorrect.** The root cause is NOT:
- Nested module structure
- Function call chain
- Complex signal dependencies

**The actual minimal trigger is simply:**
```systemverilog
module top;
  logic [7:0] data;
  always_comb data[0] = ~data[7];
endmodule
```

### Trigger Conditions

- `always_comb` block with bit-select assignment
- Target and source bits must be from the same signal
- Signal width ≥ 8 bits triggers timeout
- Signal width < 8 bits triggers crash (different bug: XorOp::canonicalize segfault)

## Additional Findings

During minimization, discovered a **separate bug**:
- With `logic [1:0] data; always_comb data[0] = ~data[1];`
- Causes segfault in `circt::comb::XorOp::canonicalize`
- This is a different bug from the timeout issue

## Final Minimized Testcase

```systemverilog
// Minimal testcase for circt-verilog --ir-hw timeout
// Bug: Compilation hangs when processing always_comb with bit-select
module top;
  logic [7:0] data;
  always_comb data[0] = ~data[7];
endmodule
```

## Reproduction Command

```bash
timeout 30s circt-verilog --ir-hw bug.sv
# Expected: Exit code 124 (timeout)
```

## Suspected Location

Based on the minimal testcase, the timeout likely occurs in:
- MooreToCore conversion when handling `always_comb`
- Specifically in handling bit-select operations with self-referencing signals
