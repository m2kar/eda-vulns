# Validation Report

## Test Case Summary

| Property | Value |
|----------|-------|
| File | `bug.sv` |
| Lines | 5 (reduced from 12, **58% reduction**) |
| Crash Type | Assertion failure |
| Reproducible | ✅ Yes |

## Minimized Test Case

```systemverilog
// Minimal test case: string type output port causes assertion failure
// Bug: CIRCT crashes when a module has string type output port
// Expected: Proper error message or correct handling
module test_module(output string str_out);
endmodule
```

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv
```

## Syntax Validation

**IEEE Standard**: 1800-2005 / 1800-2017

**Result**: ✅ Syntactically valid

SystemVerilog allows `string` type as module port per IEEE 1800. While not synthesizable, it is valid for simulation purposes.

## Cross-Tool Validation

| Tool | Version | Result | Notes |
|------|---------|--------|-------|
| **Verilator** | 5.022 | ✅ Pass | Lint-only mode passes silently |
| **Slang** | 10.0.6 | ✅ Pass | Accepts as valid syntax |
| **Icarus** | 13.0 | ⚠️ Error | Graceful "not supported" message |
| **CIRCT** | 1.139.0 | ❌ Crash | Assertion failure |

### Icarus Verilog Behavior (Expected)
```
bug.sv:4: sorry: Port `str_out` of module `test_module` with type `string` is not supported.
1 error(s) during elaboration.
```

This is the **expected behavior** - a graceful error message when a feature is not supported.

### CIRCT Behavior (Bug)
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
PLEASE submit a bug report to https://github.com/llvm/circt
Stack dump:
...
SVModuleOpConversion::matchAndRewrite MooreToCore.cpp
```

CIRCT crashes with an assertion failure instead of emitting a proper diagnostic.

## Classification

| Attribute | Value |
|-----------|-------|
| **Result** | `report` |
| **Is Genuine Bug** | ✅ Yes |
| **Bug Type** | Crash on valid input |
| **Severity** | High |

### Reasoning

1. **Syntax is valid**: Confirmed by Verilator (lint) and Slang (full parse)
2. **Feature may be unsupported**: String ports are simulation-only, not synthesizable
3. **Expected behavior**: Graceful error message (like Icarus)
4. **Actual behavior**: Crash with assertion failure
5. **Verdict**: **This is a bug** - CIRCT should handle unsupported features gracefully

## Root Cause (from analysis.json)

The crash occurs because:
1. `moore::StringType` is converted to `sim::DynamicStringType`
2. This type is not valid for HW module ports (`hw::isHWValueType()` returns false)
3. No validation exists between type conversion and port construction
4. `dyn_cast<hw::InOutType>` in `sanitizeInOut()` fails on invalid type

## Recommendation

Report this as a **genuine bug** with classification:
- **Category**: Missing validation / crash on edge case
- **Priority**: High (compiler should never crash on valid input)
- **Suggested Fix**: Add type validation in `getModulePortInfo()` to reject non-HW types with a proper diagnostic
