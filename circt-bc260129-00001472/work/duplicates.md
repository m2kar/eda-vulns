# Duplicate Check Report

## Search Results

| Query | Results |
|-------|---------|
| `arcilator StateType llhd.ref` | 0 issues |
| `arcilator inout` | 0 issues |
| `LowerState StateType` | 0 issues |
| `arcilator crash` | 1 issue (different) |
| `state type must have a known bit width` | 1 issue (exact match!) |

## Duplicate Found

### Issue #9574 - [Arc] Assertion failure when lowering inout ports in sequential logic

**URL**: https://github.com/llvm/circt/issues/9574

**Status**: Open

**Created**: 2026-02-01

**Similarity Score**: 10.0/10.0 (Exact match)

### Comparison

| Aspect | Our Bug | Issue #9574 |
|--------|---------|-------------|
| Error Message | `state type must have a known bit width; got '!llhd.ref<i1>'` | ✅ Same |
| Crash Location | `LowerState.cpp:219` | ✅ Same |
| Trigger | `inout` port | ✅ Same |
| Tool | arcilator | ✅ Same |
| Pass | LowerStatePass | ✅ Same |
| Root Cause | StateType::get() fails on llhd.ref type | ✅ Same |

### Issue #9574 Test Case

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

### Our Minimized Test Case

```systemverilog
module Bug(inout logic c);
endmodule
```

**Note**: Our test case is more minimal, showing that the crash occurs with ANY inout port, not just when used in sequential logic.

## Recommendation

**DO NOT SUBMIT** - This is a duplicate of Issue #9574.

However, the existing issue title says "inout ports in sequential logic" which is inaccurate - the crash occurs with ANY inout port, regardless of sequential logic. Consider adding a comment to #9574 with our more minimal test case:

```systemverilog
// Even simpler reproduction - no sequential logic needed:
module Bug(inout logic c);
endmodule
```
