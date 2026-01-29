# Validation Report

## Summary

| Check | Result |
|-------|--------|
| Syntax Check | ✅ valid |
| Feature Support | ✅ valid IEEE 1800 feature |
| Known Limitations | ❌ none matched |
| **Classification** | **report** |

## Syntax Validation

**Tool**: slang v10.0.6
**Status**: ✅ Valid

```
Build succeeded: 0 errors, 0 warnings
```

The test case is syntactically correct SystemVerilog according to IEEE 1800 standard.

## Feature Support Analysis

**Feature Used**: `string` type as module port

**IEEE Standard Reference**: IEEE 1800-2017 Section 6.16 - String data type

The `string` type is a valid SystemVerilog data type defined in the IEEE 1800 standard. It is a dynamic data type primarily used for:
- Simulation and verification
- Testbench code
- Display/formatting functions

**Note**: While `string` is not synthesizable to hardware, CIRCT should either:
1. Support conversion for simulation purposes, OR
2. Emit a proper diagnostic error explaining the limitation

Instead, CIRCT crashes with an assertion failure, which is incorrect behavior.

## Cross-Tool Validation

| Tool | Version | Status | Notes |
|------|---------|--------|-------|
| slang | 10.0.6 | ✅ pass | Full IEEE 1800 compliance |
| Verilator | 5.022 | ✅ pass | Lint check passed |
| Icarus Verilog | 13.0 | ⚠️ unsupported | "Port with type string is not supported" - graceful error |

### Analysis

- **slang** and **Verilator** both accept this code without errors
- **Icarus Verilog** does not support string ports but produces a **graceful error message** rather than crashing
- **CIRCT** crashes with an assertion failure, which is clearly a bug

## Classification

**Result**: `report`

**Confidence**: High

**Reasoning**:
1. The test case uses valid IEEE 1800-2017 SystemVerilog syntax
2. Two major tools (slang, Verilator) accept this code
3. Tools that don't support this feature (Icarus) produce proper error messages
4. CIRCT crashes with an internal assertion failure instead of a user-facing error
5. This represents a bug in CIRCT's error handling, not a limitation in the test case

## Recommendation

**Action**: Proceed to check for duplicates and generate bug report.

**Suggested Issue Title**: `[Moore] Crash on string type as module port - missing StringType conversion`

**Key Points for Report**:
1. Valid SystemVerilog code causes assertion failure
2. The crash occurs in MooreToCore conversion pass
3. Root cause: No type conversion rule for `moore::StringType`
4. Expected behavior: Either support the conversion or emit proper diagnostic
