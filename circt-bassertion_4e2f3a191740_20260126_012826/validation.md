# Validation Report

## Summary

| Check | Result |
|-------|--------|
| Syntax Check | ✅ Valid |
| Feature Support | ⚠️ Unsupported (but valid SV) |
| Cross-Tool Validation | ✅ 2/3 pass |
| **Classification** | **report** |

## Classification

**Result**: `report`  
**Category**: Bug  
**Reason**: Valid SystemVerilog syntax (confirmed by slang and verilator) causes assertion failure in CIRCT. The test case uses string type as module output port which is legal per IEEE 1800-2017 but triggers a crash in MooreToCore conversion.

## Syntax Validation

**Tool**: slang 10.0.6+3d7e6cd2e  
**Status**: ✅ Valid

```
Build succeeded: 0 errors, 0 warnings
```

## Feature Support Analysis

### Unsupported Features Detected

| Feature | IEEE Section | CIRCT Status |
|---------|--------------|--------------|
| string type as module port | 6.16 String data type | Crashes instead of proper error |

### Notes

The `string` data type is a valid SystemVerilog built-in type per IEEE 1800-2017 Section 6.16. While it is a simulation-only type (not synthesizable), it is legal syntax. CIRCT should either:
1. Support simulation constructs like string ports, OR
2. Emit a proper error message instead of crashing with an assertion failure

## Cross-Tool Validation

| Tool | Version | Status | Notes |
|------|---------|--------|-------|
| slang | 10.0.6+3d7e6cd2e | ✅ Pass | Build succeeded |
| verilator | 5.022 | ✅ Pass | No errors |
| iverilog | - | ❌ Error | "Port with type string is not supported" (tool limitation) |

### Analysis

- **slang** and **verilator** both accept the code as valid SystemVerilog
- **iverilog** rejects it but explicitly states "not supported" rather than "syntax error"
- This confirms the test case is **syntactically valid** but uses a feature that some tools don't fully support

## IEEE Compliance

**Standard**: IEEE 1800-2017  
**Section**: 6.16 String data type  
**Valid**: ✅ Yes

> The string data type is an ordered collection of characters. The length of a string variable is the number of characters in the collection which can have dynamic length and vary during simulation.

String is a first-class data type in SystemVerilog, and using it as a module port is syntactically valid.

## Recommendation

**Action**: Report as bug

CIRCT should handle string port types gracefully:
- Either support simulation constructs appropriately
- Or emit a proper diagnostic error message instead of crashing

The current behavior (assertion failure) is unacceptable for any input that is syntactically valid per IEEE 1800-2017.

## Test Case

```systemverilog
module test_module(output string a);
endmodule
```

## Reproduction

```bash
circt-verilog --ir-hw bug.sv
```
