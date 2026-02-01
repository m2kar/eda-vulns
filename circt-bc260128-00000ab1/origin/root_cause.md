# Root Cause Analysis Report

## Executive Summary
circt-verilog crashes with an assertion failure when processing a SystemVerilog module that uses the `string` type as a port. The crash occurs in `ModulePortInfo::sanitizeInOut()` when attempting to `dyn_cast` a null/invalid type, indicating the TypeConverter fails to convert the `string` type to a valid HW type.

## Crash Context
- **Tool/Command**: `circt-verilog --ir-hw`
- **Dialect**: Moore (SystemVerilog frontend)
- **Failing Pass**: MooreToCore conversion pass
- **Crash Type**: Assertion failure (dyn_cast on non-existent value)

## Error Analysis

### Assertion/Error Message
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Key Stack Frames
```
#17 circt::hw::ModulePortInfo::sanitizeInOut() PortImplementation.h:177
#21 getModulePortInfo(TypeConverter, SVModuleOp) MooreToCore.cpp:259
#22 SVModuleOpConversion::matchAndRewrite() MooreToCore.cpp:276
#42 MooreToCorePass::runOnOperation() MooreToCore.cpp:2571
```

## Test Case Analysis

### Code Summary
A simple SystemVerilog module with a `string` type input port (`str_in`), using string methods (`str_in.len()`) and string comparison.

### Key Constructs
- `input string str_in` - **String type as a module port** (problematic)
- `str_in.len()` - String method call
- `str_in == "test"` - String comparison

### Potentially Problematic Patterns
The `string` type in SystemVerilog is a dynamic class type that cannot be directly represented in hardware. When used as a module port, the TypeConverter likely returns a null/invalid type since there's no valid HW type mapping for `string`.

## CIRCT Source Analysis

### Crash Location
**File**: `include/circt/Dialect/HW/PortImplementation.h`
**Function**: `ModulePortInfo::sanitizeInOut()`
**Line**: 177

### Code Context
```cpp
// PortImplementation.h:175-181
void sanitizeInOut() {
  for (auto &p : ports)
    if (auto inout = dyn_cast<hw::InOutType>(p.type)) {  // LINE 177 - CRASH HERE
      p.type = inout.getElementType();
      p.dir = ModulePort::Direction::InOut;
    }
}
```

The `dyn_cast` at line 177 operates on `p.type`. When `p.type` is null (invalid/unconverted), the assertion `isPresent(Val)` fails.

### Processing Path
1. `SVModuleOpConversion::matchAndRewrite()` is called for `moore.module`
2. Calls `getModulePortInfo(typeConverter, op)` at line 276
3. In `getModulePortInfo()` (lines 234-259):
   - Iterates over module ports
   - Calls `typeConverter.convertType(port.type)` for each port (line 243)
   - For `string` type, conversion **returns null** (no valid HW type)
   - Creates `PortInfo` with null type
4. Returns `hw::ModulePortInfo(ports)` at line 258
5. `ModulePortInfo` constructor calls `sanitizeInOut()` 
6. `sanitizeInOut()` tries to `dyn_cast` on the null type → **CRASH**

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: TypeConverter fails to convert `string` type to a valid HW type, returning null. The code path does not check for null type after conversion.

**Evidence**:
- `string` is a dynamic SystemVerilog type with no direct hardware representation
- Stack trace shows crash in `sanitizeInOut()` which iterates over port types
- `dyn_cast` assertion specifically checks for "non-existent value"
- `getModulePortInfo()` directly uses `typeConverter.convertType()` result without null check (line 243)

**Mechanism**: 
1. `typeConverter.convertType(stringType)` returns `Type()` (null)
2. Null type stored in `PortInfo.type`
3. `ModulePortInfo` constructor calls `sanitizeInOut()`
4. `dyn_cast<hw::InOutType>(nullType)` triggers assertion

### Hypothesis 2 (Medium Confidence)
**Cause**: Missing type conversion pattern for Moore string type in MooreToCorePass.

**Evidence**:
- TypeConverter patterns may not cover all Moore types
- `string` is a class-based type in SystemVerilog, may lack explicit handling

**Mechanism**: If no conversion pattern exists for `moore::StringType`, the TypeConverter returns null by default.

## Suggested Fix Directions
1. **Add null check after type conversion in `getModulePortInfo()`**: Check if `portTy` is null and emit a diagnostic error for unsupported port types
2. **Add string type conversion**: Either map to a dummy type or emit a proper error diagnostic during Moore type conversion
3. **Guard in `sanitizeInOut()`**: Add `if (!p.type) continue;` before the `dyn_cast` for defensive programming

## Keywords for Issue Search
`string port` `dyn_cast non-existent` `sanitizeInOut` `MooreToCore` `TypeConverter` `string type conversion` `module port type`

## Related Files to Investigate
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Type conversion patterns and `getModulePortInfo()`
- `include/circt/Dialect/HW/PortImplementation.h` - `sanitizeInOut()` implementation
- `lib/Dialect/Moore/MooreTypes.cpp` - Moore type definitions including string
