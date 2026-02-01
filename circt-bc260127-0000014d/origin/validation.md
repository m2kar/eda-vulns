# Validation Report

## Summary

| Field | Value |
|-------|-------|
| **Result** | `report` |
| **Confidence** | 95% |
| **Reduction** | 80.9% (204 → 39 bytes) |

## Minimized Testcase

```systemverilog
module test(input string a);
endmodule
```

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv
```

## Crash Analysis

### Error Type
Assertion failure in LLVM casting during MooreToCore dialect conversion.

### Stack Trace Key Points
1. `SVModuleOpConversion::matchAndRewrite` - conversion pattern for SVModule
2. Type conversion fails for `string` type → returns null
3. `ModulePortInfo::sanitizeInOut()` calls `dyn_cast` on null type
4. Assertion fires: `detail::isPresent(Val) && "dyn_cast on a non-existent value"`

### Root Cause
The MooreToCore type converter does not handle the SystemVerilog `string` type. When a module port has `string` type, the conversion returns a null type. The `sanitizeInOut()` function then attempts to `dyn_cast` this null type, triggering an assertion failure.

## Cross-Tool Validation

| Tool | Version | Result | Notes |
|------|---------|--------|-------|
| **Verilator** | 5.022 | ✅ Accepts | No errors or warnings |
| **Slang** | 10.0.6 | ✅ Accepts | Build succeeded: 0 errors, 0 warnings |

Both reference tools accept this as valid SystemVerilog, confirming:
1. The testcase is syntactically and semantically valid
2. This is a CIRCT bug, not an invalid input

## Classification Rationale

**Report (Bug)** - High confidence because:
1. Valid SystemVerilog accepted by Verilator and Slang
2. Crashes CIRCT with assertion failure instead of graceful error
3. Minimal 2-line reproduction
4. Clear root cause: missing type conversion for `string` type

## Issues Found

None - this is a valid bug report.
