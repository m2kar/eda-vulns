# Validation Report

## Summary

| Check | Result |
|-------|--------|
| Syntax Check | ✅ valid (slang, verilator) |
| Feature Support | ⚠️ unsupported feature |
| Known Limitations | ✅ matched |
| **Classification** | **feature_request** |

## Test Case

```systemverilog
module top(input logic clk);
  logic a, b;
  assert property (@(posedge clk) a |-> b) else $error("msg");
endmodule
```

## Minimization Results

| Metric | Value |
|--------|-------|
| Original lines | 13 |
| Minimized lines | 4 |
| Reduction | **69.2%** |

## Syntax Validation

**Tool**: slang (primary), verilator (secondary)

### Slang Results
```
Build succeeded: 0 errors, 0 warnings
Exit code: 0
```

### Verilator Results
```
Exit code: 0
```

Both slang and verilator confirm the SystemVerilog syntax is **valid** per IEEE 1800-2017.

## Feature Support Analysis

**Feature detected**: Concurrent assertion with action block (else clause)

**IEEE Reference**: IEEE 1800-2017 Section 16.5 - Concurrent assertions

The construct `assert property (...) else $error(...)` is valid IEEE 1800-2017 SystemVerilog. The action block (`else $error("msg")`) specifies what happens when the assertion fails.

### CIRCT Known Limitations

**Matched**: Yes

The CIRCT ImportVerilog component explicitly states this feature is "not supported yet". The code has conditional handling:
- Assertions **without** action blocks → converted to `verif::AssertOp`
- Assertions **with** action blocks → emits "not supported yet" error

This is a **known limitation**, not a bug.

## Cross-Tool Validation

| Tool | Status | Notes |
|------|--------|-------|
| Slang | ✅ pass | Full SVA support |
| Verilator | ✅ pass | SVA lint passes |
| Icarus Verilog | ❌ error | Limited SVA support (expected) |

**Note**: Icarus Verilog's error is expected - it has known limited support for SystemVerilog Assertions (SVA). This does not indicate a problem with the test case.

## Classification

**Result**: `feature_request`

**Confidence**: High

**Reasoning**:
1. The test case is **syntactically valid** per IEEE 1800-2017 (confirmed by slang and verilator)
2. CIRCT explicitly emits "not supported yet" - indicating a **known unimplemented feature**
3. The feature (concurrent assertions with action blocks) is a **standard** SystemVerilog construct
4. Other CIRCT paths (FIRRTL → ExportVerilog) already support emitting action blocks

## Recommendation

**File as feature request** rather than bug report.

### Suggested Title
`[ImportVerilog] Support concurrent assertions with action blocks`

### Suggested Labels
- `enhancement`
- `ImportVerilog`
- `Moore`

### Workaround
Remove the action block from concurrent assertions:
```systemverilog
// Instead of:
assert property (@(posedge clk) a |-> b) else $error("msg");

// Use:
assert property (@(posedge clk) a |-> b);
```

## References

- IEEE 1800-2017 Section 16.5: Concurrent assertions
- CIRCT source: `lib/Conversion/ImportVerilog/Statements.cpp:770`
