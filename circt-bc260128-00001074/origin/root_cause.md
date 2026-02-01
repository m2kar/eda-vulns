# Root Cause Analysis Report

## Executive Summary
`circt-verilog` crashes in the MooreToCore conversion when lowering a module that declares a `string`-typed port. The crash occurs in `hw::ModulePortInfo::sanitizeInOut()` because an invalid/empty port type is passed through `getModulePortInfo()` without validation, and `sanitizeInOut()` performs an unchecked `dyn_cast<hw::InOutType>` on it.

## Crash Context
- **Tool/Command**: `circt-verilog --ir-hw source.sv`
- **Dialect**: Moore → HW
- **Failing Pass**: MooreToCore
- **Crash Type**: Assertion (`dyn_cast on a non-existent value`)

## Error Analysis

### Assertion Message
```
dyn_cast on a non-existent value
```

### Key Stack Frames (from error.txt)
```
#17 circt::hw::ModulePortInfo::sanitizeInOut() /.../PortImplementation.h:177
#21 getModulePortInfo(...) /lib/Conversion/MooreToCore/MooreToCore.cpp:259
#22 SVModuleOpConversion::matchAndRewrite(...) /lib/Conversion/MooreToCore/MooreToCore.cpp:276
```

## Test Case Analysis

### Code Summary
The test module declares a `string` input port and assigns the output `int` to `a.len()`. The key feature is a `string` port type at the module boundary.

### Key Constructs
- `string` type as a module port
- `string.len()` method call

### Potentially Problematic Pattern
Using `string` in module port type requires Moore-to-HW type conversion; unsupported/invalid conversion can propagate an empty `mlir::Type` into HW port construction.

## CIRCT Source Analysis

### Crash Location
**File**: `include/circt/Dialect/HW/PortImplementation.h`
**Function**: `ModulePortInfo::sanitizeInOut()`
**Line**: ~177

### Code Context
```cpp
void sanitizeInOut() {
  for (auto &p : ports)
    if (auto inout = dyn_cast<hw::InOutType>(p.type)) {
      p.type = inout.getElementType();
      p.dir = ModulePort::Direction::InOut;
    }
}
```

### Processing Path
1. `getModulePortInfo()` converts each SV port type using the MooreToCore `TypeConverter`.
2. The converted `portTy` is used to construct `hw::PortInfo` without validation.
3. `hw::ModulePortInfo(ports)` calls `sanitizeInOut()` internally.
4. `sanitizeInOut()` calls `dyn_cast<hw::InOutType>(p.type)` for each port.
5. For a `string` port, the conversion yields an invalid/empty type, causing `dyn_cast` to assert.

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: The MooreToCore type conversion path does not produce a valid HW-compatible type for `string` ports, returning an empty/invalid `mlir::Type`. `getModulePortInfo()` does not guard against this invalid type before constructing `ModulePortInfo`, which triggers an unchecked `dyn_cast` in `sanitizeInOut()`.

**Evidence**:
- Error log stack trace points to `sanitizeInOut()` in `PortImplementation.h:177`.
- `getModulePortInfo()` in `MooreToCore.cpp` constructs `PortInfo` from `portTy` without null/validity checks.
- `sanitizeInOut()` blindly dyn_casts port types without confirming presence.

**Mechanism**: invalid `portTy` → `PortInfo` → `ModulePortInfo` → `sanitizeInOut()` → `dyn_cast` assertion.

## Suggested Fix Directions
1. **Validate converted port types in `getModulePortInfo()`**: detect null/invalid `Type` and emit a diagnostic for unsupported string ports.
2. **Defensive guard in `sanitizeInOut()`**: skip or diagnose when `p.type` is empty before attempting `dyn_cast`.
3. **Provide an explicit conversion for `string` ports** (if supported) or reject them earlier with a user-facing error.

## Keywords for Issue Search
`string` `StringType` `MooreToCore` `getModulePortInfo` `sanitizeInOut` `dyn_cast` `InOutType` `port type` `type conversion`

## Related Files to Investigate
- `lib/Conversion/MooreToCore/MooreToCore.cpp` — `getModulePortInfo()` port type conversion
- `include/circt/Dialect/HW/PortImplementation.h` — `ModulePortInfo::sanitizeInOut()`
