# Validation Report

## Summary

| Check | Result |
|-------|--------|
| Syntax Check | ✅ valid |
| Feature Support | ✅ valid_sv_feature |
| Known Limitations | ❌ none matched |
| **Classification** | **report** |

## Syntax Validation

**Tool**: slang 10.0.6  
**Status**: ✅ Valid

```
Build succeeded: 0 errors, 0 warnings
```

The test case `module e(output string o);endmodule` is valid SystemVerilog syntax per IEEE 1800-2017.

## Feature Support Analysis

**Feature**: `string` type output port

| Aspect | Status |
|--------|--------|
| IEEE 1800-2017 | ✅ Valid construct |
| Synthesizability | ❌ Not synthesizable (simulation only) |
| CIRCT Support | ❓ Should emit error, not crash |

### IEEE Standard Reference

Per IEEE 1800-2017:
- `string` is a built-in data type (Section 6.16)
- Module ports can have any data type (Section 23.2.2)
- However, `string` is typically not synthesizable

## Cross-Tool Validation

| Tool | Version | Status | Notes |
|------|---------|--------|-------|
| slang | 10.0.6 | ✅ Pass | Syntax accepted, no errors |
| Verilator | 5.022 | ✅ Pass | Lint check passed |
| Icarus Verilog | 13.0 | ⚠️ Unsupported | "Port with type `string` is not supported" |

### Analysis

- **slang** and **Verilator**: Accept the syntax as valid SystemVerilog
- **Icarus Verilog**: Rejects with clear error message (tool limitation, not syntax error)
- **CIRCT**: Crashes with assertion failure instead of error message

## Classification

**Result**: `report`  
**Confidence**: High

### Reasoning

1. ✅ **Valid Syntax**: The test case is syntactically correct per IEEE 1800-2017
2. ✅ **Other Tools Accept**: slang and Verilator both accept this code
3. ✅ **Clear Bug Pattern**: CIRCT crashes instead of emitting a diagnostic
4. ❌ **Not a Known Limitation**: No matching issue in CIRCT's known limitations

### Expected Behavior vs Actual

| Aspect | Expected | Actual |
|--------|----------|--------|
| Compilation | Error message or success | Assertion failure (crash) |
| Error Handling | Graceful rejection | Segmentation fault |
| User Experience | Actionable diagnostic | Stack trace |

## Recommendation

**Proceed to report as Bug.**

The issue is clear:
- Valid SystemVerilog input causes CIRCT to crash
- Proper behavior: emit error like "string type not supported for module ports"
- Actual behavior: assertion failure in MooreToCore conversion

This is a missing null-check or unsupported type handling issue in the Moore to Core conversion pass.

## Files Validated

| File | Size | Status |
|------|------|--------|
| `bug.sv` | 35 bytes | ✅ Minimized, triggers crash |
| `error.log` | ~3KB | ✅ Contains crash stack trace |
| `command.txt` | 28 bytes | ✅ Single-line reproduction |
