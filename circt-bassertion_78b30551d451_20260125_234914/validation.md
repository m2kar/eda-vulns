# Validation Report

## Summary

| Check | Result |
|-------|--------|
| Syntax Check | valid |
| Feature Support | supported |
| Known Limitations | none |
| **Classification** | **report** |

## Syntax Validation

**Tool**: slang
**Status**: valid
**Message**: Build succeeded: 0 errors, 0 warnings

The test case is valid IEEE 1800-2017 SystemVerilog.

## Feature Support Analysis

**Unsupported features detected**: None

The `string` type is a valid SystemVerilog data type defined in IEEE 1800-2017. Using it as a module port is syntactically valid.

### CIRCT Known Limitations

No known limitation matched.

## Cross-Tool Validation

| Tool | Status | Notes |
|------|--------|-------|
| Slang | pass | Build succeeded |
| Verilator | pass | No errors |
| Icarus | error | "Net `a` can not be of type `string`" - Icarus limitation, not SV standard violation |

**Analysis**: 2 out of 3 tools accept this code. Icarus Verilog has limited SystemVerilog support and rejects string port types, but this is an Icarus limitation, not a problem with the test case.

## Classification

**Result**: `report`

**Reasoning**:
The test case is valid SystemVerilog code accepted by industry-standard tools (slang, verilator). CIRCT crashes instead of:
1. Successfully processing the code, OR
2. Emitting a proper diagnostic error if the feature is not supported

This is a genuine bug - compiler crashes are never acceptable behavior.

## Recommendation

Proceed to check for duplicates and generate the bug report. The crash represents a missing validation or unhandled case in the MooreToCore conversion pass.

## Bug Nature

According to `analysis.json`:
- **Root Cause**: `hw::ModulePortInfo::sanitizeInOut()` assumes all port types are HW dialect compatible
- **Mechanism**: `dyn_cast<hw::InOutType>` on `sim::DynamicStringType` triggers assertion failure
- **Suggested Fix**: Add port type validation before conversion, emit diagnostic for unsupported types
