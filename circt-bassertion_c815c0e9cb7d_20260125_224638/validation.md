# Validation Report

## Summary

| Check | Result |
|-------|--------|
| Syntax Check | ✅ valid |
| Feature Support | ⚠️ valid_but_limited |
| Known Limitations | none matched |
| **Classification** | **report** |

## Syntax Validation

**Tool**: slang  
**Status**: ✅ valid

```
Build succeeded: 0 errors, 0 warnings
```

The test case is valid IEEE 1800-2017 SystemVerilog. The `string` data type as a module port is syntactically correct per IEEE 1800-2017 Section 6.16.

## Feature Support Analysis

**Feature**: `string` type as module port

### IEEE Standard Reference

IEEE 1800-2017 defines the `string` data type in Section 6.16. While `string` is primarily intended for simulation and testbenches, the syntax for using it as a module port is valid.

### CIRCT Known Limitations

No known limitation matched in our database.

**Note**: This appears to be a new crash that should be reported.

## Cross-Tool Validation

| Tool | Status | Notes |
|------|--------|-------|
| slang | ✅ pass | Full IEEE 1800-2017 compliance - syntax accepted |
| Verilator | ✅ pass | Accepts string port declaration (lint-only) |
| Icarus Verilog | ❌ error | "Port with type `string` is not supported" |

### Analysis

- **slang**: Reference implementation, confirms syntax is valid IEEE 1800
- **Verilator**: Also accepts the syntax (lint mode)
- **Icarus**: Explicitly reports "not supported" with a clear error message

The key observation is that **Icarus Verilog provides a proper error message** when it doesn't support a feature, while **CIRCT crashes with an assertion failure**. This is the core bug.

## Classification

**Result**: `report`

**Reasoning**:
1. The test case is syntactically valid IEEE 1800-2017 SystemVerilog
2. slang (reference parser) and Verilator accept the syntax
3. When a tool doesn't support a feature, it should emit a proper diagnostic (like Icarus does)
4. CIRCT crashes with `dyn_cast on a non-existent value` instead of providing helpful feedback
5. Even if `string` ports are not supported, CIRCT should fail gracefully

## Recommendation

**Proceed to report as a bug.**

The issue is that CIRCT crashes when encountering `string`-typed module ports during MooreToCore conversion. The expected behavior should be either:

1. **Support the feature**: Convert string ports to appropriate simulation types
2. **Reject gracefully**: Emit a clear error like "string type ports are not supported"

An assertion failure is never acceptable user-facing behavior.

## Test Case

```systemverilog
module top_module(output string str_out);
endmodule
```

**Reproduction**:
```bash
circt-verilog --ir-hw bug.sv
```
