# Minimization Report

## Summary
- **Original file**: source.sv (22 lines)
- **Minimized file**: bug.sv (2 lines)
- **Reduction**: 90.9%
- **Crash preserved**: Yes

## Preservation Analysis

### Key Constructs Preserved
Based on `analysis.json`, the following constructs were kept:
- `string` type output port (`output string str_out`)

### Removed Elements
- `always_ff` block (not essential for crash)
- `always_comb` block (not essential for crash)
- `assign` statement (not essential for crash)
- Internal `string` variable (not essential for crash)
- `input` ports (not essential for crash)
- `output logic O` port (not essential for crash)

## Verification

### Original Assertion
```
dyn_cast on a non-existent value
```

### Final Assertion
```
(anonymous namespace)::SVModuleOpConversion::matchAndRewrite crash in MooreToCore.cpp
```

**Match**: ✅ Same crash location (SVModuleOpConversion in MooreToCore)

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv
```

## Notes
- The crash is triggered by the `string` type port alone
- No internal logic or other ports are needed
- This confirms the root cause: MooreToCore lacks StringType conversion
