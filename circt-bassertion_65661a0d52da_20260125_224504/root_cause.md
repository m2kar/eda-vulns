# Root Cause Analysis Report

## Executive Summary

CIRCT's `circt-verilog` tool crashes with an assertion failure when processing a SystemVerilog module that uses the `string` type as an output port. The crash occurs in the `MooreToCorePass` during module port type conversion, where `dyn_cast<hw::InOutType>` is called on a potentially null/invalid type. The root cause is that the `getModulePortInfo()` function does not check if type conversion succeeds before using the converted type.

## Crash Context

- **Tool/Command**: `circt-verilog --ir-hw source.sv`
- **Dialect**: Moore → HW/LLHD conversion
- **Failing Pass**: `MooreToCorePass` (`ConvertMooreToCore`)
- **Crash Type**: Assertion failure

## Error Analysis

### Assertion/Error Message

```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

This assertion is from LLVM's `Casting.h:650`, indicating that `dyn_cast` was called on a null/invalid type value.

### Key Stack Frames

```
#11 mlir::TypeStorage::getAbstractType()
#12 mlir::Type::getTypeID()
#13-16 CastInfo<circt::hw::InOutType, mlir::Type>::isPossible()
#17 circt::hw::ModulePortInfo::sanitizeInOut() [PortImplementation.h:177]
#18-21 (anonymous namespace)::getModulePortInfo() [MooreToCore.cpp:259]
#22 SVModuleOpConversion::matchAndRewrite() [MooreToCore.cpp:276]
#35 MooreToCorePass::runOnOperation() [MooreToCore.cpp:2571]
```

## Test Case Analysis

### Code Summary

The test case defines a SystemVerilog module `Mod` with:
- An input port `a` of type `logic [1:0]`
- An **output port `str_out` of type `string`**
- An internal string variable and a combinational block that assigns string values

A top module `Top` instantiates `Mod` and connects the string output to a local string variable.

### Key Constructs

1. **`string` type as module output port** - The critical construct causing the crash
2. String literal assignment (`"test"`, `"default"`)
3. String variable declaration and assignment
4. Module instantiation with string port connection

### Potentially Problematic Patterns

The use of `string` type (an unpacked, dynamically-sized type) as a module port is the problematic pattern. While SystemVerilog allows this, the CIRCT HW dialect may not fully support non-synthesizable types as module ports.

## CIRCT Source Analysis

### Crash Location

**File**: `lib/Conversion/MooreToCore/MooreToCore.cpp:259` (returning from `getModulePortInfo()`)
**Function**: `sanitizeInOut()` in `include/circt/Dialect/HW/PortImplementation.h:177`

### Code Context

**`getModulePortInfo()` (MooreToCore.cpp:233-259)**:
```cpp
static hw::ModulePortInfo getModulePortInfo(const TypeConverter &typeConverter,
                                            SVModuleOp op) {
  // ...
  for (auto port : moduleTy.getPorts()) {
    Type portTy = typeConverter.convertType(port.type);  // Line 243
    // NOTE: No null check on portTy!
    if (port.dir == hw::ModulePort::Direction::Output) {
      ports.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, resultNum++, {}}));
    } else {
      ports.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, inputNum++, {}}));
    }
  }
  return hw::ModulePortInfo(ports);  // Line 258 - triggers sanitizeInOut()
}
```

**`sanitizeInOut()` (PortImplementation.h:175-181)**:
```cpp
void sanitizeInOut() {
  for (auto &p : ports)
    if (auto inout = dyn_cast<hw::InOutType>(p.type)) {  // Line 177 - CRASH
      p.type = inout.getElementType();
      p.dir = ModulePort::Direction::InOut;
    }
}
```

**Type conversion for `StringType` (MooreToCore.cpp:2277-2279)**:
```cpp
typeConverter.addConversion([&](StringType type) {
  return sim::DynamicStringType::get(type.getContext());
});
```

### Processing Path

1. `MooreToCorePass::runOnOperation()` starts conversion
2. `SVModuleOpConversion::matchAndRewrite()` converts `moore.module`
3. Calls `getModulePortInfo()` to get HW port information
4. For `str_out` port with `!moore.string` type:
   - `typeConverter.convertType(port.type)` is called
   - The type converter maps `StringType` → `sim::DynamicStringType`
   - **However**, the returned type may be null if conversion fails
5. `hw::ModulePortInfo` constructor calls `sanitizeInOut()`
6. `dyn_cast<hw::InOutType>` is called on potentially null type
7. **CRASH**: Assertion failure in `dyn_cast`

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)

**Cause**: Missing null check after `typeConverter.convertType()` in `getModulePortInfo()`

**Evidence**:
- The assertion `dyn_cast on a non-existent value` indicates a null/invalid type
- Line 243 calls `typeConverter.convertType()` without checking the result
- MLIR's `TypeConverter::convertType()` returns `nullptr` on conversion failure
- The `sim::DynamicStringType` may not be accepted as a valid HW module port type

**Mechanism**:
1. `StringType` has a registered converter to `sim::DynamicStringType`
2. However, `sim::DynamicStringType` is not a valid type for HW module ports
3. The conversion may succeed, but subsequent operations fail
4. OR: The type conversion fails silently, returning null
5. The null type propagates to `sanitizeInOut()` where `dyn_cast` fails

### Hypothesis 2 (Medium Confidence)

**Cause**: `sim::DynamicStringType` is not supported as an HW module port type

**Evidence**:
- `hw::isHWValueType()` only recognizes: `IntegerType`, `IntType`, `EnumType`, `ArrayType`, `StructType`, `UnionType`, `TypeAliasType`
- `sim::DynamicStringType` is NOT in this list
- HW module ports traditionally require synthesizable types

**Mechanism**:
Even if type conversion "succeeds", the resulting `sim::DynamicStringType` may not be usable in the HW dialect context, leading to downstream failures.

### Hypothesis 3 (Low Confidence)

**Cause**: Type converter registration order issue

**Evidence**:
- Type converters are registered in `populateTypeConversion()`
- The order of `addConversion()` calls matters in MLIR's TypeConverter
- A more general converter might intercept `StringType` before the specific one

**Mechanism**:
A catch-all or overly general converter might handle `StringType` incorrectly, returning an incompatible or null type.

## Suggested Fix Directions

1. **Immediate fix**: Add null check in `getModulePortInfo()`:
   ```cpp
   Type portTy = typeConverter.convertType(port.type);
   if (!portTy) {
     // Emit diagnostic error about unsupported port type
     return failure();  // Or appropriate error handling
   }
   ```

2. **Proper fix**: Either:
   - Emit a proper diagnostic error when `string` type is used as a module port
   - Or implement proper lowering of `string` ports (if desired behavior)

3. **Alternative**: Mark modules with `string` ports as illegal during legality checking

## Keywords for Issue Search

`string` `StringType` `DynamicStringType` `MooreToCore` `getModulePortInfo` `sanitizeInOut` `dyn_cast` `port` `type conversion` `assertion` `InOutType`

## Related Files to Investigate

| File | Reason |
|------|--------|
| `lib/Conversion/MooreToCore/MooreToCore.cpp` | Main conversion pass, crash location |
| `include/circt/Dialect/HW/PortImplementation.h` | `sanitizeInOut()` implementation |
| `include/circt/Dialect/HW/HWTypes.h` | `isHWValueType()` definition |
| `include/circt/Dialect/Sim/SimTypes.td` | `DynamicStringType` definition |
| `include/circt/Dialect/Moore/MooreTypes.td` | `StringType` definition |
