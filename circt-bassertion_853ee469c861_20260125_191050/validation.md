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
**Status**: ✅ valid

```
Build succeeded: 0 errors, 0 warnings
```

The test case uses valid IEEE 1800-2017 SystemVerilog syntax. The `string` type as a module port is explicitly supported by the standard.

## Feature Support Analysis

**Unsupported features detected**: None

### IEEE 1800 Reference

From IEEE 1800-2017:
- Section 6.16: String data type
- Section 23.2.2: Port declarations

The `string` type is a built-in data type that can be used in port declarations for simulation purposes.

### CIRCT Known Limitations

No known limitation matched for this specific case. The crash indicates a missing validation rather than a documented unsupported feature.

## Cross-Tool Validation

| Tool | Status | Exit Code | Notes |
|------|--------|-----------|-------|
| Slang | ✅ pass | 0 | Build succeeded |
| Verilator | ✅ pass | 0 | Lint passed |
| Icarus | ❌ error | 1 | Tool limitation (not testcase issue) |

### Slang Output
```
Build succeeded: 0 errors, 0 warnings
```

### Verilator Output
```
(no errors)
```

### Icarus Output
```
bug.sv:1: sorry: Port `str_out` of module `test` with type `string` is not supported.
```

**Note**: Icarus Verilog does not support `string` type ports, but this is a limitation of Icarus, not the test case. Both Slang (IEEE reference parser) and Verilator accept the code.

## Classification

**Result**: `report`  
**Confidence**: High

### Reasoning

1. **Valid SystemVerilog**: The test case passes Slang (IEEE reference parser) and Verilator validation
2. **Unique CIRCT crash**: Other tools either accept the code or reject with proper error messages
3. **Assertion failure**: CIRCT crashes with an assertion instead of providing a diagnostic
4. **Should be reported**: Even if string ports aren't supported in HW synthesis flow, CIRCT should emit a proper error, not crash

### Expected Behavior

CIRCT should either:
1. **Support** string type ports (convert to appropriate simulation type), OR
2. **Reject** with a clear error message like "string type ports are not supported for hardware synthesis"

The current behavior (assertion failure) is incorrect.

## Recommendation

✅ **Proceed to check for duplicates and generate bug report.**

This is a genuine bug in CIRCT's MooreToCore conversion pass. The crash occurs because:
1. SystemVerilog `string` type is parsed successfully by Moore dialect
2. Type converter maps it to `sim::DynamicStringType`
3. HW module port creation fails because `DynamicStringType` is not a valid HW type
4. Missing validation causes an assertion failure instead of proper error handling

## Test Case Quality

| Metric | Value |
|--------|-------|
| Original lines | 24 |
| Minimized lines | 2 |
| Reduction | 91.7% |
| Crash reproducible | Yes |
| Syntax valid | Yes |
| Cross-tool validated | Yes (2/3 pass) |

The minimized test case is high quality: minimal, valid, and clearly demonstrates the bug.
