# Validation Report

## Test Case
```systemverilog
module test(output string result);
endmodule
```

## Summary
| Aspect | Result |
|--------|--------|
| **Language** | SystemVerilog |
| **Syntax Valid** | ✅ Yes |
| **IEEE Compliant** | ✅ Yes (IEEE 1800-2017) |
| **Classification** | `valid_testcase` |
| **Is Genuine Bug** | ✅ Yes |

## Cross-Tool Validation

### Slang (Reference Parser)
- **Status**: ✅ PASSED
- **Output**: `Build succeeded: 0 errors, 0 warnings`
- **Significance**: Slang is a highly compliant SystemVerilog parser. Its acceptance confirms the syntax is valid.

### Verilator
- **Status**: ✅ PASSED (with warnings)
- **Warnings**:
  - `DECLFILENAME`: Filename doesn't match module name (cosmetic)
  - `UNDRIVEN`: Output signal not driven (expected for minimal test)
- **Exit Code**: 0
- **Significance**: Verilator accepts the syntax as valid SystemVerilog.

### Icarus Verilog
- **Status**: ❌ FAILED
- **Error**: `Port 'result' of module 'test' with type 'string' is not supported`
- **Significance**: Icarus has limited SystemVerilog support. This failure is expected and does not indicate invalid syntax.

### CIRCT (circt-verilog)
- **Status**: 💥 CRASH
- **Type**: Assertion failure
- **Location**: `MooreToCore.cpp:SVModuleOpConversion`
- **Significance**: This is the bug being reported.

## IEEE 1800 Analysis

### String Type (Section 6.16)
The `string` data type is defined in IEEE 1800-2017 Section 6.16:
- String is a **dynamic data type** (variable-length)
- It is valid to use string as a port type in SystemVerilog
- However, string is **not synthesizable** (simulation-only construct)

### Port Declaration (Section 23.2)
Module ports can have any valid data type, including `string`.

## Classification Rationale

**Result: `valid_testcase` - Genuine Bug**

1. **Syntax is valid**: Confirmed by Slang (0 errors, 0 warnings)
2. **IEEE compliant**: String type ports are legal per IEEE 1800
3. **Crash is unacceptable**: Even if CIRCT doesn't support string ports for synthesis, it should:
   - Emit a proper diagnostic error message
   - NOT crash with an assertion failure

## Recommendation

**Action**: Report as bug to CIRCT

**Rationale**:
- Compiler crashes on valid input are always bugs
- The fix should either:
  1. Add proper support for string type ports, OR
  2. Emit a clear error message like "string type ports are not supported for hardware synthesis"

**Severity**: High (crash on valid input)
