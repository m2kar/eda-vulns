# Root Cause Analysis Report

## Crash Summary
- **Crash Type**: Assertion Failure
- **Testcase ID**: 260129-00001473
- **Dialect**: MooreToCore Conversion
- **Assertion**: `detail::isPresent(Val) && "dyn_cast on a non-existent value"`

## Crash Location
- **File**: `lib/Conversion/MooreToCore/MooreToCore.cpp:259`
- **Function**: `getModulePortInfo()`
- **Call Chain**: 
  1. `MooreToCorePass::runOnOperation()`
  2. `SVModuleOpConversion::matchAndRewrite()`
  3. `getModulePortInfo()`
  4. `hw::ModulePortInfo::sanitizeInOut()`

## Source Code Analysis

### Triggering Input (`source.sv`)
```systemverilog
module test_module(
  input logic clk,
  input logic rst,
  output logic [1:0] sel,
  output string s          // <-- PROBLEMATIC: string type as module port
);
  // ...
endmodule
```

### Critical Code Path

**1. `getModulePortInfo()` in MooreToCore.cpp:234-259:**
```cpp
static hw::ModulePortInfo getModulePortInfo(const TypeConverter &typeConverter,
                                            SVModuleOp op) {
  // ...
  for (auto port : moduleTy.getPorts()) {
    Type portTy = typeConverter.convertType(port.type);  // Line 243
    // ... create PortInfo with portTy ...
  }
  return hw::ModulePortInfo(ports);  // Line 258 - triggers sanitizeInOut()
}
```

**2. `sanitizeInOut()` in PortImplementation.h:175-181:**
```cpp
void sanitizeInOut() {
  for (auto &p : ports)
    if (auto inout = dyn_cast<hw::InOutType>(p.type)) {  // CRASHES HERE
      p.type = inout.getElementType();
      p.dir = ModulePort::Direction::InOut;
    }
}
```

## Root Cause

### Primary Issue: Missing Null Check After Type Conversion

When converting a Moore `StringType` port to HW dialect, the type converter at line 2277-2278 converts it to `sim::DynamicStringType`:

```cpp
typeConverter.addConversion([&](StringType type) {
  return sim::DynamicStringType::get(type.getContext());
});
```

However, `sim::DynamicStringType` is **NOT** a valid type for hardware module ports in the HW dialect. When `hw::ModulePortInfo` is constructed with this port, the `sanitizeInOut()` method is called which attempts to `dyn_cast<hw::InOutType>` on ALL port types.

The crash occurs because:
1. `dyn_cast<hw::InOutType>()` receives a `sim::DynamicStringType` 
2. The `sim::DynamicStringType` may have an invalid/null internal state when accessed as `hw::InOutType`
3. The LLVM casting infrastructure asserts `detail::isPresent(Val)` fails

### Secondary Issue: Semantic Mismatch

The `string` type in SystemVerilog is a **dynamic string** (similar to `std::string` in C++). It is:
- Not synthesizable
- Not a valid hardware port type
- Only meaningful in simulation context

The MooreToCore conversion should:
1. Either reject `string` type ports with a proper error diagnostic
2. Or properly handle this case in `getModulePortInfo()` by skipping such ports or emitting appropriate errors

## Affected Components

| Component | File | Issue |
|-----------|------|-------|
| TypeConverter | MooreToCore.cpp:2277 | Converts StringType to sim::DynamicStringType |
| getModulePortInfo | MooreToCore.cpp:243 | No null/validity check on converted port type |
| sanitizeInOut | PortImplementation.h:177 | Assumes all port types can be safely cast |

## Suggested Fix

### Option 1: Validate Port Types in getModulePortInfo()
```cpp
for (auto port : moduleTy.getPorts()) {
  Type portTy = typeConverter.convertType(port.type);
  if (!portTy || !hw::isHWValueType(portTy)) {
    // Emit error: unsupported port type
    return failure();
  }
  // ... proceed with valid type
}
```

### Option 2: Guard in sanitizeInOut()
```cpp
void sanitizeInOut() {
  for (auto &p : ports)
    if (p.type && isa<hw::InOutType>(p.type)) {
      auto inout = cast<hw::InOutType>(p.type);
      // ...
    }
}
```

### Option 3: Reject StringType Ports Early
Add validation in MooreToCorePass to reject modules with string-type ports with clear diagnostics.

## Impact Assessment

- **Severity**: High (crash/assertion failure)
- **Reproducibility**: 100% with string-type module ports
- **Affected Versions**: CIRCT 1.139.0 (and likely earlier versions)
- **User Impact**: Users cannot process SystemVerilog with string-type ports

## Minimal Reproduction

```bash
echo 'module m(output string s); endmodule' | circt-verilog --ir-hw -
```

## References

- Stack Frame #17: `circt::hw::ModulePortInfo::sanitizeInOut()`
- Stack Frame #21: `getModulePortInfo()` at MooreToCore.cpp:259
- Stack Frame #22: `SVModuleOpConversion::matchAndRewrite()` at MooreToCore.cpp:276
