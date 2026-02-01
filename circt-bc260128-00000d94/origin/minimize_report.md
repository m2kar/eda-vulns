# Minimization Report

## Summary

| Metric | Value |
|--------|-------|
| **Original file** | source.sv (21 lines) |
| **Minimized file** | bug.sv (4 lines) |
| **Reduction** | 81% |
| **Crash preserved** | Yes |

## Key Constructs Preserved

Based on `analysis.json`, the following construct was essential:
- `output string out_str` - String type as module output port

## Removed Elements

| Element | Lines Removed | Reason |
|---------|---------------|--------|
| Input port `signed_data` | 1 | Not needed for crash |
| Internal `logic sel` | 1 | Not needed for crash |
| String array `s[0:0]` | 1 | Not needed for crash |
| First `always_comb` block | 3 | Not needed for crash |
| Second `always_comb` block | 5 | Not needed for crash |
| `assign` statement | 1 | Not needed for crash |
| Empty lines | 4 | Cleanup |

## Minimized Test Case

```systemverilog
module test_module(
  output string out_str
);
endmodule
```

## Verification

### Original Crash Signature
```
SVModuleOpConversion::matchAndRewrite() crashes during MooreToCore conversion
Assertion: dyn_cast on a non-existent value (null type from StringType conversion)
```

### Minimized Crash Signature
```
SVModuleOpConversion::matchAndRewrite at MooreToCore.cpp
Stack trace shows same conversion path: MooreToCorePass → SVModuleOpConversion
```

**Match**: ✅ Same crash location and root cause

## Root Cause Analysis

The crash occurs because:
1. `StringType` (dynamic string) cannot be converted to a hardware type
2. `typeConverter.convertType()` returns null for `StringType` ports
3. `getModulePortInfo()` passes the null type to `ModulePortInfo`
4. `sanitizeInOut()` crashes when calling `dyn_cast<InOutType>()` on null

The minimal test case with just `output string out_str` triggers this exact path.

## Reproduction Command

```bash
/opt/firtool/bin/circt-verilog --ir-hw bug.sv
```

## Notes

- The original test case had additional logic (comparisons, array, always blocks) that was not related to the crash
- Only the `string` output port declaration is necessary to trigger the bug
- The crash is in the Moore-to-Core dialect conversion, specifically in port type handling
