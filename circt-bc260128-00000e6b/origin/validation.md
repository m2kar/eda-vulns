# Validation Report

## Summary

| Check | Result |
|-------|--------|
| Syntax Check | valid (slang) |
| Feature Support | supported |
| Known Limitations | none |
| **Classification** | **report** |

## Minimization

| Metric | Value |
|--------|-------|
| Original lines | 9 |
| Minimized lines | 2 |
| Reduction | 77.8% |

### Minimized Test Case

```systemverilog
module test(input string a);
endmodule
```

## Syntax Validation

**Tool**: slang
**Status**: valid

```
Build succeeded: 0 errors, 0 warnings
```

## Cross-Tool Validation

| Tool | Status | Notes |
|------|--------|-------|
| Slang | ✓ pass | Syntax valid |
| Verilator | ✓ pass | Lint passed |
| Icarus | ✗ error | `Net 'a' can not be of type 'string'` - iverilog limitation |

### Analysis

- **Slang** (IEEE 1800-2017 compliant): Accepts the code
- **Verilator**: Accepts the code
- **Icarus Verilog**: Rejects - but iverilog has known limitations with SystemVerilog types

The test case uses valid IEEE 1800-2017 SystemVerilog syntax. The `string` type is a valid data type per the standard.

## Classification

**Result**: `report`

**Reasoning**:
The test case is valid SystemVerilog code accepted by Slang (the most standards-compliant parser) and Verilator. CIRCT crashes instead of either:
1. Supporting the construct
2. Emitting a clear error message

This is a bug - compiler crashes are never acceptable behavior for valid input.

## Root Cause (from analysis.json)

When converting a Moore module with string-type ports to HW dialect:
1. The type converter produces `sim::DynamicStringType`
2. `hw::ModulePortInfo::sanitizeInOut()` performs `dyn_cast<hw::InOutType>` on all port types
3. The dyn_cast fails with an assertion for incompatible `sim::DynamicStringType`

## Recommendation

**Proceed to check for duplicates and generate the bug report.**

The crash occurs in MooreToCore pass during module port type conversion. CIRCT should either:
- Support string ports properly
- Emit a diagnostic error for unsupported port types before conversion
