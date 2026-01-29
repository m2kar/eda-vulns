# Root Cause Analysis Report

## Executive Summary

CIRCT crashes when processing a SystemVerilog module with `string` type output ports. The MooreToCore conversion pass correctly converts `moore::StringType` to `sim::DynamicStringType`, but this type is not a valid HW value type for module ports. When `ModulePortInfo::sanitizeInOut()` is called with this invalid type, `dyn_cast<hw::InOutType>` fails with assertion because `sim::DynamicStringType` is not recognized as a valid port type.

## Crash Context

- **Tool**: circt-verilog (version 1.139.0)
- **Dialect**: Moore
- **Failing Pass**: MooreToCore (Moore dialect to Core/HW conversion)
- **Crash Type**: Assertion failure
- **Hash**: 6397c80f76a4

## Error Analysis

### Assertion Message
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed
```

### Key Stack Frames
```
#11 MooreToCore.cpp:259 - getModulePortInfo()
#12 MooreToCore.cpp:276 - SVModuleOpConversion::matchAndRewrite()
```

### Crash Location
- **File**: `lib/Conversion/MooreToCore/MooreToCore.cpp`
- **Line**: 259 (end of `getModulePortInfo()` function)
- **Function**: `getModulePortInfo(const TypeConverter&, SVModuleOp)`

## Test Case Analysis

### Code Summary
```systemverilog
module test_module(output string str_out, output logic [7:0] count_out);
  string a = "Test";
  logic [7:0] count;
  assign str_out = a;
  assign count_out = count;
  always @(a.len()) begin
    count <= 8'd0;
  end
endmodule
```

### Key Constructs
1. `output string str_out` - **String type as module output port** (problematic)
2. `string a = "Test"` - String variable declaration
3. `always @(a.len())` - String method call in sensitivity list (secondary issue)

### Problematic Pattern
The module declares a `string` type output port. While CIRCT's type converter can convert `moore::StringType` to `sim::DynamicStringType`, this converted type is **not a valid HW value type** for hardware module ports.

## CIRCT Source Analysis

### Crash Mechanism

1. **Type Conversion (MooreToCore.cpp:243)**:
   ```cpp
   Type portTy = typeConverter.convertType(port.type);
   // For string type: moore::StringType -> sim::DynamicStringType
   ```

2. **Port Info Construction (MooreToCore.cpp:246-254)**:
   ```cpp
   ports.push_back(hw::PortInfo({{port.name, portTy, port.dir}, resultNum++, {}}));
   // portTy = sim::DynamicStringType (non-null, but not a valid HW type)
   ```

3. **ModulePortInfo Constructor (PortImplementation.h)**:
   ```cpp
   explicit ModulePortInfo(ArrayRef<PortInfo> mergedPorts)
       : ports(mergedPorts.begin(), mergedPorts.end()) {
     sanitizeInOut();  // <-- Crash here
   }
   ```

4. **sanitizeInOut() (PortImplementation.h)**:
   ```cpp
   void sanitizeInOut() {
     for (auto &p : ports)
       if (auto inout = dyn_cast<hw::InOutType>(p.type)) {
         // dyn_cast fails because sim::DynamicStringType is not castable
         // to hw::InOutType, but more critically, the type system
         // doesn't handle this gracefully
   ```

### The Real Issue

The type `sim::DynamicStringType` fails validation in HW dialect context:
- `hw::isHWValueType(sim::DynamicStringType)` returns `false`
- HW module ports must be "HW value types" (integer, array, struct, etc.)
- String types are simulation-only types, not synthesizable

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence): Missing Port Type Validation

**Cause**: `getModulePortInfo()` doesn't validate that converted port types are valid HW value types.

**Evidence**:
- `typeConverter.convertType()` returns `sim::DynamicStringType` for string ports
- `sim::DynamicStringType` is not a valid HW value type (checked by `hw::isHWValueType()`)
- No validation between type conversion and port construction
- The assertion fails in `sanitizeInOut()` when processing invalid port types

**Mechanism**:
```
StringType (Moore) 
  -> sim::DynamicStringType (converted)
  -> PortInfo constructed (no validation)
  -> ModulePortInfo::sanitizeInOut() called
  -> dyn_cast<hw::InOutType> on non-HW type
  -> ASSERTION FAILURE
```

**Fix Direction**:
Add validation in `getModulePortInfo()`:
```cpp
Type portTy = typeConverter.convertType(port.type);
if (!portTy || !hw::isHWValueType(portTy)) {
  // Emit proper error instead of crashing
  return failure();
}
```

### Hypothesis 2 (Medium Confidence): String Port Not Supported

**Cause**: String type module ports are not supported in hardware synthesis, but there's no graceful error.

**Evidence**:
- SystemVerilog `string` is a dynamic/simulation type
- HW dialect only supports synthesizable types
- CIRCT should reject string ports at an earlier stage with a meaningful error

**Fix Direction**:
Add validation during Moore dialect parsing or import to reject string ports with a diagnostic message.

### Hypothesis 3 (Low Confidence): Type Converter Issue

**Cause**: The type converter should return `nullptr` or fail for unsupported port types.

**Evidence**:
- Type converters can return `nullptr` to indicate conversion failure
- The current converter returns a valid type (`sim::DynamicStringType`) that is later invalid in context

## Suggested Fix Directions

1. **Immediate Fix** (in `getModulePortInfo()`):
   ```cpp
   Type portTy = typeConverter.convertType(port.type);
   if (!portTy) {
     op.emitError("unsupported port type: ") << port.type;
     return hw::ModulePortInfo({});  // or return failure
   }
   if (!hw::isHWValueType(portTy)) {
     op.emitError("port type not supported in HW modules: ") << portTy;
     return hw::ModulePortInfo({});
   }
   ```

2. **Better Fix** (at Moore import stage):
   Reject string type ports during Slang import with a clear error message.

3. **Long-term Fix**:
   Consider if string ports should be supported for simulation-only flows (non-synthesizable paths).

## Keywords for Issue Search

`string` `port` `DynamicStringType` `isHWValueType` `MooreToCore` `ModulePortInfo` `sanitizeInOut` `dyn_cast` `assertion`

## Related Files

- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Crash location
- `include/circt/Dialect/HW/PortImplementation.h` - `sanitizeInOut()` definition
- `lib/Dialect/HW/HWTypes.cpp` - `isHWValueType()` implementation
- `include/circt/Dialect/Sim/SimTypes.td` - `DynamicStringType` definition
