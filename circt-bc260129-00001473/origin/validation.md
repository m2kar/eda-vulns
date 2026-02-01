# Validation Report

## Summary

| Field | Value |
|-------|-------|
| Testcase ID | 260129-00001473 |
| Validation Result | **REPORT** |
| Reduction | 92.0% (25 → 2 lines) |

## Minimized Testcase

```systemverilog
module m(output string s);
endmodule
```

## Syntax Validation

| Tool | Status | Errors | Warnings |
|------|--------|--------|----------|
| slang | ✅ PASS | 0 | 0 |
| verilator | ✅ PASS | 0 | 0 |

The testcase is **valid SystemVerilog syntax**.

## Crash Verification

| Property | Value |
|----------|-------|
| Reproducible | ✅ Yes |
| Tool | circt-verilog |
| Version | CIRCT 1.139.0 |
| Crash Type | Assertion Failure |
| Signature Match | ✅ Yes |

### Command
```bash
circt-verilog --ir-hw bug.sv
```

### Assertion Message
```
detail::isPresent(Val) && "dyn_cast on a non-existent value"
```

## Root Cause Analysis

- **Component**: MooreToCore conversion pass
- **Functions**: `getModulePortInfo()` / `sanitizeInOut()`
- **Issue**: String type port is converted to `sim::DynamicStringType` which is not a valid type for `hw::ModulePortInfo`
- **Trigger**: `output string s` in module port list

## Classification

| Category | Value |
|----------|-------|
| Bug Type | Type conversion validation |
| Valid Bug | ✅ Yes |
| User Error | ❌ No |
| Synthesizable | ❌ No (but valid SV) |

## Recommendation

**Action: REPORT**

This is a valid compiler bug. The input is syntactically correct SystemVerilog (verified by slang and verilator). The compiler should either:
1. Successfully lower the string type port, or
2. Emit a diagnostic error message

Instead, it crashes with an assertion failure, which is unacceptable behavior for a compiler.

## Files Generated

- `bug.sv` - Minimized testcase
- `error.log` - Crash output
- `command.txt` - Reproduction command
- `validation.json` - Structured validation data
