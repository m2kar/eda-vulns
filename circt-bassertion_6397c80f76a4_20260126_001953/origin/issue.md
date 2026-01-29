# [MooreToCore] Crash on string type module output port

## Description

CIRCT crashes with an assertion failure when processing a SystemVerilog module that declares a `string` type output port. The crash occurs in the MooreToCore conversion pass at `ModulePortInfo::sanitizeInOut()`.

## Reproduction

### Minimal Test Case

```systemverilog
// Minimal test case: string type output port causes assertion failure
// Bug: CIRCT crashes when a module has string type output port
// Expected: Proper error message or correct handling
module test_module(output string str_out);
endmodule

```

### Command to Reproduce

```bash
circt-verilog --ir-hw bug.sv
```

### Expected Behavior

Either:
1. Proper error message: "string type ports are not supported in HW modules"
2. Correct handling if string ports should be supported

### Actual Behavior

```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed
```

Stack trace:
```
#0 0x... llvm::sys::PrintStackTrace(...)
...
#11 ... MooreToCore.cpp:259 - getModulePortInfo()
#12 ... MooreToCore.cpp:276 - SVModuleOpConversion::matchAndRewrite()
```

## Crash Details

| Field | Value |
|-------|-------|
| **Crash Type** | Assertion failure |
| **Failing Pass** | MooreToCore |
| **Dialect** | Moore |
| **Crash Location** | `lib/Conversion/MooreToCore/MooreToCore.cpp:259` |
| **Function** | `getModulePortInfo()` |
| **Assertion** | `dyn_cast on a non-existent value` |

## Root Cause Analysis

The type converter correctly converts `moore::StringType` to `sim::DynamicStringType`, but this type is not a valid HW value type for module ports. When `getModulePortInfo()` constructs a `ModulePortInfo` without validating the port type, `sanitizeInOut()` is called with an invalid type, causing the assertion failure.

### Mechanism

```
moore::StringType (port)
  → typeConverter.convertType()
  → sim::DynamicStringType (non-null, but not a valid HW type)
  → PortInfo constructed (no validation)
  → ModulePortInfo::sanitizeInOut()
  → dyn_cast<hw::InOutType> on invalid type
  → ASSERTION FAILURE
```

### Key Issue

The function `getModulePortInfo()` in `MooreToCore.cpp` does not validate that converted port types are valid HW value types (`hw::isHWValueType()` returns `false` for `sim::DynamicStringType`).

## Cross-Tool Validation

| Tool | Version | Result | Notes |
|------|---------|--------|-------|
| Verilator | 5.022 | ✅ Pass | Syntax is valid |
| Slang | 10.0.6 | ✅ Pass | Syntax is valid |
| Icarus Verilog | 13.0 | ⚠️ Error | "not supported" (graceful error) |
| **CIRCT** | **1.139.0** | ❌ **Crash** | **Assertion failure** |

**Conclusion**: The test case is syntactically valid per IEEE 1800-2005, but string ports are not synthesizable. CIRCT should emit a proper error message (like Icarus) instead of crashing.

## Duplicate Check

- **Recommendation**: likely_new
- **Top Score**: 7.5
- **Closest Issue**: #8930 (same assertion, different trigger)

No exact duplicates found. Related issues:
- #8930: Same assertion in MooreToCore but triggered by sqrt/floor
- #8332: Feature request for StringType support in Moore

## Suggested Fix

Add validation in `getModulePortInfo()` before constructing port info:

```cpp
Type portTy = typeConverter.convertType(port.type);
if (!portTy || !hw::isHWValueType(portTy)) {
  op.emitError("unsupported port type in HW modules: ") << port.type;
  return hw::ModulePortInfo({});
}
```

Alternative: Reject string type ports during Moore dialect parsing/import with a diagnostic message.

## Additional Context

- Test case reduced from 12 to 5 lines (58% reduction)
- Bug is reproducible on CIRCT version 1.139.0
- Crash hash: `dyn_cast on a non-existent value'

## Keywords

`string, port, DynamicStringType, isHWValueType, MooreToCore, ModulePortInfo, sanitizeInOut, dyn_cast, assertion`
