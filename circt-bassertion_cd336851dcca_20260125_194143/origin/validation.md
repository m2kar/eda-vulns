# Validation Report

## Summary

| Item | Value |
|------|-------|
| Classification | **report** (valid bug) |
| Confidence | high |
| Syntax Valid | Yes |
| Cross-tool Agreement | Yes (slang/verilator pass, CIRCT crashes) |

## Test Case

```systemverilog
module test_module(output string result);
endmodule
```

## Syntax Validation

### slang (v10.0.6)
- **Result**: PASS
- **Errors**: 0
- **Warnings**: 0
- **Output**: `Build succeeded: 0 errors, 0 warnings`

### Verilator (v5.022)
- **Result**: PASS (with non-fatal warnings)
- **Errors**: 0
- **Warnings**: 2 (DECLFILENAME, UNDRIVEN)
- **Notes**: Warnings are cosmetic and do not indicate invalid syntax

## Cross-Tool Validation

| Tool | Result | Notes |
|------|--------|-------|
| slang | Pass | Valid SystemVerilog |
| Verilator | Pass | Valid (warnings are lint, not syntax) |
| CIRCT | **Crash** | Assertion failure in MooreToCore |

## Minimization Results

| Metric | Value |
|--------|-------|
| Original lines | 13 |
| Minimized lines | 2 |
| Reduction | **84.6%** |

## Verdict

**This is a valid bug report.**

- The test case uses valid IEEE 1800 SystemVerilog syntax
- Both slang and Verilator accept the code without errors
- CIRCT crashes with an assertion failure
- The bug is in MooreToCore pass: missing type converter for `string` type on module ports

## Root Cause

The `string` type is a valid SystemVerilog data type that can be used in module ports for simulation/testbench purposes. CIRCT's MooreToCore conversion pass lacks a type converter for `moore::StringType`, causing `typeConverter.convertType()` to return null. This null type propagates to `hw::PortInfo` and eventually triggers the assertion `dyn_cast on a non-existent value` in `ModulePortInfo::sanitizeInOut()`.

## Recommendation

Submit bug report to CIRCT with:
1. Minimal reproducer (2 lines)
2. Clear assertion message and stack trace
3. Suggested fix: Add StringType converter or emit proper error diagnostic
