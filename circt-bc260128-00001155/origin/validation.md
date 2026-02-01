# Validation Report

## Summary

| Check | Result |
|-------|--------|
| Syntax Check | valid |
| Feature Support | supported |
| Known Limitations | none |
| **Classification** | **report** |

## Minimization Results

| Metric | Value |
|--------|-------|
| Original Lines | 9 |
| Minimized Lines | 2 |
| Reduction | **77.8%** |

### Minimized Test Case (bug.sv)

```systemverilog
module M(input string a);
endmodule
```

## Syntax Validation

**Tool**: slang  
**Status**: valid

```
Build succeeded: 0 errors, 0 warnings
```

## Cross-Tool Validation

| Tool | Status | Notes |
|------|--------|-------|
| Slang | ✅ pass | 0 errors, 0 warnings |
| Verilator | ✅ pass | Exit 0 |
| Icarus Verilog | ❌ error | Net `a` can not be of type `string` (iverilog limitation) |

### Analysis

- **Slang** and **Verilator** both accept this as valid IEEE 1800 SystemVerilog
- **Icarus Verilog** rejects it, but this is a known limitation of iverilog (doesn't support string port type)
- The `string` type as module port is valid per IEEE 1800-2017 standard

## Classification

**Result**: `report`

**Reasoning**:  
This test case uses valid IEEE 1800 SystemVerilog syntax that is accepted by major industry tools (slang, verilator). The crash in CIRCT is a bug - the tool should either:
1. Support string type ports correctly, or
2. Emit a proper diagnostic error instead of crashing

A compiler crash on valid input is always a bug, regardless of feature support status.

## Root Cause Summary

From `analysis.json`:
- **Dialect**: Moore
- **Failing Pass**: MooreToCore
- **Issue**: Missing type conversion rule for `moore::StringType` in `populateTypeConversion()`
- **Mechanism**: StringType port → convertType returns null → null stored in PortInfo → sanitizeInOut dyn_cast fails

## Recommendation

**Proceed to check for duplicates and generate the bug report.**

This is a valid bug that should be reported to CIRCT. The crash occurs due to missing implementation for string type conversion, not due to invalid input.
