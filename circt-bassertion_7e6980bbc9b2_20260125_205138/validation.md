# Validation Report

## Summary

| Check | Result |
|-------|--------|
| Syntax Check | ✅ valid |
| Feature Support | ✅ valid IEEE 1800 SystemVerilog |
| Known Limitations | None matched |
| **Classification** | **report** |

## Syntax Validation

**Tool**: slang
**Status**: valid

```
Build succeeded: 0 errors, 0 warnings
```

The test case `bug.sv` is syntactically valid SystemVerilog according to IEEE 1800 standard.

## Feature Support Analysis

**Feature used**: `string` type as module port

According to IEEE 1800 (SystemVerilog standard):
- `string` is a valid SystemVerilog data type
- Module ports can be of `string` type
- This is a valid language construct

**CIRCT Status**: The `string` type port causes a crash in MooreToCore type conversion. CIRCT should either:
1. Support `string` type ports, or
2. Emit a clear error message for unsupported types (not crash)

## Cross-Tool Validation

| Tool | Status | Notes |
|------|--------|-------|
| Slang | ✅ pass | Build succeeded: 0 errors, 0 warnings |
| Verilator | ✅ pass | No errors or warnings |
| Icarus Verilog | ⚠️ unsupported | "Port with type `string` is not supported" |

**Analysis**: 
- 2 out of 3 tools (slang, verilator) accept the test case
- Icarus Verilog explicitly reports "not supported" (graceful handling)
- CIRCT crashes with assertion failure (ungraceful handling)

## Classification

**Result**: `report`

**Reasoning**:
The test case is valid SystemVerilog code that is accepted by mainstream EDA tools (slang, verilator). While `string` type ports may not be fully synthesizable, CIRCT should either:
1. Support the feature, or
2. Emit a clear error message indicating the feature is not supported

The current behavior (assertion failure crash) is a bug regardless of whether the feature is supported.

## Recommendation

**Proceed to check for duplicates and generate the bug report.**

The bug should be categorized as:
- **Severity**: Medium (crash, but not on typical RTL code)
- **Type**: Missing type conversion handler causes assertion failure
- **Impact**: Any SystemVerilog code using `string` type ports will crash CIRCT
