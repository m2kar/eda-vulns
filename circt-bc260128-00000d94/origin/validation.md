# Validation Report

## Summary

| Check | Result |
|-------|--------|
| Syntax Check | ✅ valid |
| Feature Support | ✅ supported |
| Known Limitations | ❌ none matched |
| **Classification** | **report** |

## Syntax Validation

**Tool**: slang
**Status**: valid
**Output**: `Build succeeded: 0 errors, 0 warnings`

The test case is valid IEEE 1800 SystemVerilog.

## Feature Support Analysis

**Unsupported features detected**: None

The `string` type is defined in IEEE 1800-2017 (and earlier). It is a valid SystemVerilog data type. While it's a "dynamic" type not suitable for synthesis, it should not cause a compiler crash.

### CIRCT Known Limitations

No known limitation matched. This appears to be a previously unreported issue.

## Cross-Tool Validation

| Tool | Status | Notes |
|------|--------|-------|
| Verilator | ✅ pass | No errors (lint-only mode) |
| Slang | ✅ pass | Build succeeded: 0 errors, 0 warnings |
| Icarus Verilog | ⚠️ not_supported | "sorry: Port with type `string` is not supported" |

**Interpretation**: 
- Verilator and Slang both accept the code as valid SystemVerilog
- Icarus Verilog reports it as unsupported (their limitation), but this is different from a syntax error
- The test case is valid IEEE 1800 SystemVerilog

## Classification

**Result**: `report`

**Confidence**: high

**Reasoning**:
The test case is valid SystemVerilog code that causes a crash in CIRCT's Moore-to-Core conversion pass. The crash occurs because:

1. `StringType` ports cannot be converted to hardware types
2. The type converter returns null for `StringType`
3. Missing null check causes crash in `sanitizeInOut()`

This is a bug because:
- Valid input should not cause a compiler crash
- Even if `string` ports are unsupported, CIRCT should emit an error message, not crash
- Both Verilator and Slang accept this code without crashing

## Recommendation

**Proceed to check for duplicates and generate the bug report.**

The crash should be reported to CIRCT. The fix would be to add a null check after `typeConverter.convertType()` in `getModulePortInfo()` and emit a proper diagnostic if the type cannot be converted.

## Test Case

```systemverilog
module test_module(
  output string out_str
);
endmodule
```

## Reproduction Command

```bash
/opt/firtool/bin/circt-verilog --ir-hw bug.sv
```
