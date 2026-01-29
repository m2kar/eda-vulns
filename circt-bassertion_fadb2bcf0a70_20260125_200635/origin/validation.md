# Validation Report

## Summary
| Item | Value |
|------|-------|
| **Result** | `report` (Valid Bug Report) |
| **Syntax Valid** | ✅ Yes |
| **Cross-Tool Verified** | ✅ Yes |
| **Confidence** | High |

## Test Case
```systemverilog
module example(
  output string str
);
endmodule
```

## Syntax Validation

The test case uses valid SystemVerilog syntax according to IEEE 1800-2017:
- `string` is a built-in data type (IEEE 1800-2017 §6.16)
- Using `string` as a module port is syntactically valid

## Cross-Tool Validation

| Tool | Version | Result | Errors | Warnings |
|------|---------|--------|--------|----------|
| Verilator | 5.022 | ✅ Pass | 0 | 0 |
| Slang | 10.0.6 | ✅ Pass | 0 | 0 |
| CIRCT | 1.139.0 | ❌ Crash | - | - |

### Verilator Output
```
(no output - lint passed)
```

### Slang Output
```
Build succeeded: 0 errors, 0 warnings
```

### CIRCT Output
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

## Classification

### Is this a valid test case?
**Yes** - The code follows IEEE 1800-2017 SystemVerilog standard.

### Is this a genuine bug?
**Yes** - CIRCT crashes with an assertion failure instead of:
1. Successfully processing the code, or
2. Emitting a proper diagnostic if the feature is unsupported

### Is this an unsupported feature?
**Partially** - String type ports may not be fully supported in CIRCT's hardware synthesis flow, but:
- The tool should emit a clear error message, not crash
- Assertion failures indicate missing validation, not intentional limitation

### Is this a design limitation?
**No** - This is a crash bug, not a documented limitation.

## Recommendation

**Report this as a bug** because:
1. Valid SystemVerilog code causes a crash
2. Other tools (Verilator, Slang) handle it correctly
3. Even if string ports are unsupported, CIRCT should fail gracefully with a diagnostic

## Severity
**High** - Assertion failures can cause unexpected tool termination and poor user experience.

## Root Cause (from analysis)
The `sim::DynamicStringType` (converted from Moore `StringType`) is not a valid HW port type. The `ModulePortInfo::sanitizeInOut()` function calls `dyn_cast<hw::InOutType>` without checking if the type is valid, causing the assertion failure.
