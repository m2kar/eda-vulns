# Root Cause Analysis Report

## Executive Summary
**circt-verilog** crashes with an assertion failure when converting a Moore dialect module with `string` type output port to hw dialect. The `sim::DynamicStringType` (converted from Moore `StringType`) is passed to `hw::ModulePortInfo` constructor, which calls `dyn_cast<hw::InOutType>` on a non-hw type, triggering the assertion "dyn_cast on a non-existent value".

## Crash Context
- **Tool**: circt-verilog
- **Dialect**: Moore
- **Failing Pass**: MooreToCore (convert-moore-to-core)
- **Crash Type**: Assertion failure
- **CIRCT Version**: 1.139.0

## Error Analysis

### Assertion Message
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Crash Location
- **File**: `lib/Conversion/MooreToCore/MooreToCore.cpp:259`
- **Function**: `getModulePortInfo()`
- **Context**: `hw::ModulePortInfo(ports)` constructor call

### Key Stack Frames
```
#11 0x0000557bbbeaeb57 (assertion in dyn_cast)
#12 getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp)
    MooreToCore.cpp:259
#13 SVModuleOpConversion::matchAndRewrite(...)
    MooreToCore.cpp:276
#35 MooreToCorePass::runOnOperation()
    MooreToCore.cpp:2571
```

## Test Case Analysis

### Code Summary
```systemverilog
module example_module(
  input logic clk,
  output string out  // <-- problematic: string type as output port
);
  string str;
  
  always_comb begin
    str = "test";
    out = str;
  end
endmodule
```

### Key Constructs
- Module with `string` type output port
- `string` type variable declaration
- `always_comb` block with string assignment

### Problematic Pattern
The SystemVerilog `string` type is used as a module output port. While Moore dialect can represent this, the conversion to hw dialect fails because `string` types are not valid hw port types.

## CIRCT Source Analysis

### Type Conversion Path
1. **Moore StringType** → converted by `typeConverter.addConversion([&](StringType type) {...})` at line 2277
2. **Result**: `sim::DynamicStringType` (a sim dialect type, not hw type)

### Crash Mechanism
1. `getModulePortInfo()` (line 234-259) iterates over module ports
2. For each port, `typeConverter.convertType(port.type)` converts Moore types to target types
3. `StringType` converts to `sim::DynamicStringType`
4. `hw::ModulePortInfo(ports)` is constructed (line 258)
5. Constructor calls `sanitizeInOut()` (PortImplementation.h:175-181)
6. `sanitizeInOut()` calls `dyn_cast<hw::InOutType>(p.type)` for each port
7. **CRASH**: `sim::DynamicStringType` is not an hw type, `dyn_cast` fails on incompatible type

### Code Context
**MooreToCore.cpp:242-258**:
```cpp
for (auto port : moduleTy.getPorts()) {
  Type portTy = typeConverter.convertType(port.type);  // StringType -> sim::DynamicStringType
  if (port.dir == hw::ModulePort::Direction::Output) {
    ports.push_back(
        hw::PortInfo({{port.name, portTy, port.dir}, resultNum++, {}}));
  } else {
    ports.push_back(
        hw::PortInfo({{port.name, portTy, port.dir}, inputNum++, {}}));
  }
}

return hw::ModulePortInfo(ports);  // CRASH: sanitizeInOut() calls dyn_cast on non-hw type
```

**PortImplementation.h:175-181**:
```cpp
void sanitizeInOut() {
  for (auto &p : ports)
    if (auto inout = dyn_cast<hw::InOutType>(p.type)) {  // ASSERTION FAILS HERE
      p.type = inout.getElementType();
      p.dir = ModulePort::Direction::InOut;
    }
}
```

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: Missing validation for hw-compatible port types in `getModulePortInfo()`

**Evidence**:
- `StringType` converts to `sim::DynamicStringType`, a non-hw type
- `hw::ModulePortInfo` constructor assumes all port types are hw-compatible
- No check exists for whether converted type is suitable as hw port

**Mechanism**: 
The type conversion pipeline correctly converts `StringType` to `sim::DynamicStringType`, but `getModulePortInfo()` doesn't validate that the converted type is compatible with hw module ports. When the incompatible type reaches `sanitizeInOut()`, the `dyn_cast` fails because `sim::DynamicStringType` cannot be checked against `hw::InOutType`.

### Hypothesis 2 (Medium Confidence)
**Cause**: `string` type module ports are not supported but lack proper error handling

**Evidence**:
- `string` is a dynamic type not suitable for hardware synthesis
- hw dialect only supports synthesizable types for ports
- No diagnostic message is emitted before the crash

**Mechanism**:
Using `string` as module port is likely an unsupported feature that should produce a proper error message rather than an assertion failure.

### Hypothesis 3 (Low Confidence)
**Cause**: `sanitizeInOut()` implementation is not robust against non-hw types

**Evidence**:
- `dyn_cast` should use `dyn_cast_if_present` or check for valid type first
- The function assumes all port types are hw dialect types

**Mechanism**:
Even if string ports are unsupported, the code should handle non-hw types gracefully rather than crashing.

## Suggested Fix Directions

1. **Add port type validation in `getModulePortInfo()`**:
   ```cpp
   Type portTy = typeConverter.convertType(port.type);
   if (!portTy || !hw::isHWValueType(portTy)) {
     // Emit error diagnostic and return failure
     return failure();
   }
   ```

2. **Make `sanitizeInOut()` robust**:
   ```cpp
   void sanitizeInOut() {
     for (auto &p : ports)
       if (p.type && isa<hw::InOutType>(p.type))
         if (auto inout = dyn_cast<hw::InOutType>(p.type)) {
           p.type = inout.getElementType();
           p.dir = ModulePort::Direction::InOut;
         }
   }
   ```

3. **Add unsupported feature error** for string port types in Moore frontend:
   - Detect string ports early and emit proper diagnostic
   - "string type not supported as module port"

## Keywords for Issue Search
`string` `StringType` `DynamicStringType` `ModulePortInfo` `sanitizeInOut` `dyn_cast` `InOutType` `MooreToCore` `port` `type conversion`

## Related Files
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Type conversion and module conversion
- `include/circt/Dialect/HW/PortImplementation.h` - `sanitizeInOut()` function
- `include/circt/Dialect/Sim/SimTypes.td` - `DynamicStringType` definition
- `include/circt/Dialect/HW/HWTypes.h` - hw type utilities
