# Root Cause Analysis Report

## Executive Summary

`circt-verilog` crashes with an assertion failure when processing a SystemVerilog module that has a `string` type output port. The crash occurs during the MooreToCore conversion pass when constructing HW module port information. The `getModulePortInfo` function fails to properly handle cases where type conversion returns an invalid/empty type, causing a `dyn_cast` assertion failure in the `ModulePortInfo::sanitizeInOut()` method.

## Crash Context

- **Tool/Command**: `circt-verilog --ir-hw`
- **Dialect**: Moore (SystemVerilog frontend)
- **Failing Pass**: MooreToCore conversion
- **Crash Type**: Assertion failure

## Error Analysis

### Assertion/Error Message
```
circt-verilog: llvm/llvm/include/llvm/Support/Casting.h:650: 
decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]: 
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Key Stack Frames
```
#11 MooreToCore.cpp (assertion handler)
#12 llvm::SmallVector<circt::hw::PortInfo, 1u>::~SmallVector()
#13 getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp) MooreToCore.cpp:259
#14 SVModuleOpConversion::matchAndRewrite(...) MooreToCore.cpp:276
#35 MooreToCorePass::runOnOperation() MooreToCore.cpp:2571
```

## Test Case Analysis

### Code Summary
The test case defines a module `test_module` with:
- An input port `arg0` of type `logic [31:0]`
- An output port `a` of type `string`
- A local unpacked array `arr [0:3]` of `logic [31:0]`
- Combinational logic that conditionally assigns string values to `a`
- A local `int` variable assigned from a string method `a.len()`

### Key Constructs
- **`output string a`**: A string-typed output port - the problematic construct
- **`a = "test"` / `a = ""`**: String literal assignments
- **`a.len()`**: String method call

### Potentially Problematic Patterns
The use of `string` as a module port type is the triggering pattern. In SystemVerilog, `string` is a dynamic data type intended for simulation/verification, not synthesis. The HW dialect (which targets hardware synthesis) does not support dynamic string types as module ports.

## CIRCT Source Analysis

### Crash Location
**File**: `lib/Conversion/MooreToCore/MooreToCore.cpp`
**Function**: `getModulePortInfo()`
**Line**: ~259 (function exit, destructor of SmallVector triggers the issue)

### Code Context
```cpp
// MooreToCore.cpp:233-259
static hw::ModulePortInfo getModulePortInfo(const TypeConverter &typeConverter,
                                            SVModuleOp op) {
  size_t inputNum = 0;
  size_t resultNum = 0;
  auto moduleTy = op.getModuleType();
  SmallVector<hw::PortInfo> ports;
  ports.reserve(moduleTy.getNumPorts());

  for (auto port : moduleTy.getPorts()) {
    Type portTy = typeConverter.convertType(port.type);  // <-- May return empty Type!
    if (port.dir == hw::ModulePort::Direction::Output) {
      ports.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, resultNum++, {}}));  // <-- Stores empty Type
    } else {
      ports.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, inputNum++, {}}));
    }
  }

  return hw::ModulePortInfo(ports);  // <-- Constructor calls sanitizeInOut()
}
```

The `ModulePortInfo` constructor calls `sanitizeInOut()`:
```cpp
// PortImplementation.h:175-181
void sanitizeInOut() {
  for (auto &p : ports)
    if (auto inout = dyn_cast<hw::InOutType>(p.type)) {  // <-- Crashes if p.type is empty!
      p.type = inout.getElementType();
      p.dir = ModulePort::Direction::InOut;
    }
}
```

### Processing Path
1. `circt-verilog` parses the SystemVerilog file and creates Moore dialect IR
2. The module has an output port of type `moore::StringType`
3. MooreToCorePass initiates conversion
4. `SVModuleOpConversion::matchAndRewrite()` is called
5. `getModulePortInfo()` is invoked to convert port types
6. `typeConverter.convertType(port.type)` is called for the string port
7. The conversion returns an invalid/empty Type (conversion failure)
8. The empty Type is stored in `hw::PortInfo`
9. `hw::ModulePortInfo(ports)` constructor calls `sanitizeInOut()`
10. `dyn_cast<hw::InOutType>(p.type)` is called on the empty Type
11. **CRASH**: Assertion `detail::isPresent(Val)` fails

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: Missing validation of type conversion result in `getModulePortInfo()`

**Evidence**:
- The code at line 243 does `Type portTy = typeConverter.convertType(port.type)` without checking if the result is valid
- The assertion message indicates `dyn_cast` was called on a "non-existent value"
- The crash occurs in `sanitizeInOut()` which iterates over all port types
- The test case uses `string` as a port type, which may not be properly convertible to HW types

**Mechanism**: When `typeConverter.convertType()` fails to convert a type (returns `std::nullopt` internally), MLIR's TypeConverter API returns an empty `Type()`. This empty type is then passed to `dyn_cast<>` which asserts that the value must be present.

### Hypothesis 2 (Medium Confidence)
**Cause**: String type not properly supported as module port type in MooreToCore conversion

**Evidence**:
- `moore::StringType` is converted to `sim::DynamicStringType` (line 2277-2278)
- `sim::DynamicStringType` is marked as a valid target type (line 2352)
- However, HW module ports may have additional restrictions beyond basic type conversion
- The HW dialect is designed for synthesis, where dynamic strings are not meaningful

**Mechanism**: While the type itself converts successfully, the converted type (`sim::DynamicStringType`) may not be valid for use as an HW module port, causing downstream issues in port handling.

### Hypothesis 3 (Lower Confidence)
**Cause**: Race condition or incorrect order of type conversions

**Evidence**:
- The type converter has many registered conversions
- Some conversions depend on recursive calls to `convertType()`
- The order of type conversion registration might affect results

**Mechanism**: If the port type is somehow in an intermediate state during conversion, the type converter might fail to find a matching conversion and return empty.

## Suggested Fix Directions

1. **Add validation in `getModulePortInfo()`**: Check if `typeConverter.convertType()` returns a valid type before using it. If conversion fails, emit a proper error diagnostic instead of crashing.

```cpp
Type portTy = typeConverter.convertType(port.type);
if (!portTy) {
  // Emit diagnostic or return failure
  return failure();  // or emit error
}
```

2. **Add test for string-typed ports**: If string ports are not supported for HW conversion, add an early check and emit a user-friendly error message.

3. **Fix `sanitizeInOut()` defensive programming**: Add a null check before calling `dyn_cast`:

```cpp
void sanitizeInOut() {
  for (auto &p : ports)
    if (p.type && isa<hw::InOutType>(p.type)) {  // Check p.type first
      auto inout = cast<hw::InOutType>(p.type);
      p.type = inout.getElementType();
      p.dir = ModulePort::Direction::InOut;
    }
}
```

## Keywords for Issue Search
`string` `port` `MooreToCore` `getModulePortInfo` `dyn_cast` `InOutType` `type conversion` `assertion` `sanitizeInOut` `DynamicStringType`

## Related Files to Investigate
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Main conversion pass, crash location
- `include/circt/Dialect/HW/PortImplementation.h` - ModulePortInfo and sanitizeInOut()
- `include/circt/Dialect/Moore/MooreTypes.h` - StringType definition
- `include/circt/Dialect/Sim/SimTypes.td` - DynamicStringType definition
