# Validation Report

## Summary

| Check | Result |
|-------|--------|
| Syntax Check | valid |
| Feature Support | supported |
| Known Limitations | none |
| **Classification** | **report** |

## Minimization

| Metric | Value |
|--------|-------|
| Original Lines | 12 |
| Minimized Lines | 2 |
| **Reduction** | **83.3%** |

### Minimized Test Case

```systemverilog
module t(output string s);
endmodule
```

## Syntax Validation

**Tool**: slang
**Status**: valid ✅

```
Build succeeded: 0 errors, 0 warnings
```

## Cross-Tool Validation

| Tool | Status | Notes |
|------|--------|-------|
| slang | ✅ pass | Syntax valid |
| verilator | ✅ pass | Lint passes |
| icarus | ⚠️ unsupported | "Port with type `string` is not supported" |

### Analysis

The test case uses **valid SystemVerilog syntax** according to IEEE 1800-2017:
- `string` is a standard SystemVerilog data type
- Using `string` as a module port is syntactically legal

Both **slang** and **verilator** accept this code without errors. Icarus Verilog reports it as unsupported (not a syntax error), which is a legitimate limitation for a subset implementation.

## Classification

**Result**: `report`

**Reasoning**:
The test case is **valid SystemVerilog** that causes a **crash** in CIRCT. Even if `string` ports are not yet supported by CIRCT, the tool should:
1. Emit a proper diagnostic error message
2. Exit gracefully with a non-zero status

Instead, CIRCT crashes with an assertion failure:
```
dyn_cast on a non-existent value
```

This is a **bug** - missing error handling in the type conversion path.

## Recommendation

**Proceed to check for duplicates and generate the bug report.**

The root cause (from analysis.json) is that `moore::StringType` lacks a type converter in the MooreToCore pass, leading to a null type being passed to downstream processing.
