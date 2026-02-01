# Validation Report

## Summary

| Check | Result |
|-------|--------|
| Syntax Check (slang) | valid |
| Feature Support | supported |
| Known Limitations | none |
| **Classification** | **report** |

## Syntax Validation

**Tool**: slang

```
Build succeeded: 0 errors, 0 warnings
```

## Feature Support Analysis

No explicitly unsupported SystemVerilog features detected. The test uses a `string` port and `len()` method.

## Cross-Tool Validation

| Tool | Status | Notes |
|------|--------|-------|
| Verilator | pass | No diagnostics emitted |
| Icarus | error | `Net 'a' can not be of type 'string'` |
| Slang | pass | Lint-only succeeded |

## Classification

**Result**: `report`

**Reasoning**: The minimized test case is valid per slang and Verilator. CIRCT crashes during MooreToCore conversion, while Icarus Verilog lacks string port support and is treated as a tool limitation.
