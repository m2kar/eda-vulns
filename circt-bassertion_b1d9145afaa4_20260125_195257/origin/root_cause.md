# Root Cause Analysis Report

## Executive Summary

The crash occurs when `circt-verilog` attempts to convert a Moore dialect module with a `string` type output port to the HW dialect. The `string` type is converted to `sim::DynamicStringType`, which is not a valid hardware port type. The `getModulePortInfo` function then calls `typeConverter.convertType()` on the port type, but `sim::DynamicStringType` cannot be used in `hw::ModulePortInfo` as it's not a synthesizable hardware type. The assertion `dyn_cast<InOutType>` fails because `DynamicStringType` is not an `InOutType`.

## Crash Context

- **Tool**: circt-verilog
- **Dialect**: Moore
- **Failing Pass**: MooreToCorePass
- **Crash Type**: Assertion failure
- **Hash**: b1d9145afaa4

## Error Analysis

### Assertion Message
```
circt-verilog: llvm/include/llvm/Support/Casting.h:650: 
decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]: 
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Key Stack Frames
```
#13 getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp)
    /lib/Conversion/MooreToCore/MooreToCore.cpp:259
#14 SVModuleOpConversion::matchAndRewrite(...)
    /lib/Conversion/MooreToCore/MooreToCore.cpp:276
#35 MooreToCorePass::runOnOperation()
    /lib/Conversion/MooreToCore/MooreToCore.cpp:2571
```

### Crash Location
- **File**: lib/Conversion/MooreToCore/MooreToCore.cpp
- **Line**: 259
- **Function**: `getModulePortInfo`

## Test Case Analysis

### Code Summary
```systemverilog
module test_module(input logic clk, input logic rst_n, output string out_str);
  string a = "Test";
  initial out_str = "";
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) out_str <= "";
    else out_str <= a;
  end
endmodule
```

### Key Constructs
1. **`string` type as module output port** - The critical construct causing the crash
2. **`string` variable declaration** - Internal string variable `a`
3. **String literal assignments** - `""` and `"Test"`
4. **`always_ff` block with string assignment** - Sequential logic assigning strings

### Problematic Patterns
- Using `string` (a simulation-only data type per IEEE 1800) as a module port
- `string` is a dynamic, variable-length type that cannot be synthesized to hardware

## CIRCT Source Analysis

### Crash Location Analysis

The crash occurs in `getModulePortInfo()` at line 259:

```cpp
static hw::ModulePortInfo getModulePortInfo(const TypeConverter &typeConverter,
                                            SVModuleOp op) {
  // ...
  for (auto port : moduleTy.getPorts()) {
    Type portTy = typeConverter.convertType(port.type);  // <-- Problem here
    if (port.dir == hw::ModulePort::Direction::Output) {
      ports.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, resultNum++, {}}));
    }
    // ...
  }
  return hw::ModulePortInfo(ports);
}
```

### Type Conversion Chain

1. **Moore dialect**: `moore::StringType` (represents SystemVerilog `string`)
2. **Type conversion**: `moore::StringType` → `sim::DynamicStringType`
   ```cpp
   typeConverter.addConversion([&](StringType type) {
     return sim::DynamicStringType::get(type.getContext());
   });
   ```
3. **HW module creation**: `hw::HWModuleOp::create()` receives port with `sim::DynamicStringType`
4. **Assertion failure**: When validating/processing ports, code expects hardware-compatible types

### Processing Path
1. Parse SystemVerilog → Create Moore dialect IR
2. `SVModuleOp` created with `string` output port
3. `MooreToCorePass` triggers conversion
4. `SVModuleOpConversion::matchAndRewrite()` called
5. `getModulePortInfo()` converts port types
6. `string` → `sim::DynamicStringType` (valid conversion)
7. Attempt to create `hw::PortInfo` with non-hardware type
8. Internal validation/cast fails (assertion)

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: Missing validation for unsynthesizable types used as module ports before MooreToCore conversion

**Evidence**:
- `string` is converted to `sim::DynamicStringType` which is a simulation-only type
- `hw::ModulePortInfo` is designed for synthesizable hardware types only
- The type converter successfully converts `string` → `DynamicStringType`, but this type is invalid for HW module ports
- No early rejection of non-synthesizable port types

**Mechanism**: 
The type conversion system allows `string` to be converted to `sim::DynamicStringType`, but the HW dialect doesn't support this type for module ports. The code assumes all converted port types are valid hardware types, leading to an assertion failure when processing the port info.

### Hypothesis 2 (Medium Confidence)
**Cause**: `typeConverter.convertType()` returns a non-null but invalid type for hardware module ports

**Evidence**:
- The assertion is `dyn_cast on a non-existent value`
- This suggests `convertType()` might return `nullptr` or an invalid type
- `sim::DynamicStringType` is valid for simulation but not for hardware synthesis

**Mechanism**:
The type converter should return `std::nullopt` or emit an error for types that cannot be used as hardware module ports, instead of successfully converting to a simulation-only type.

### Hypothesis 3 (Low Confidence)
**Cause**: Missing `sim::DynamicStringType` handling in `hw::PortInfo` construction

**Evidence**:
- The HW dialect has `InOutType` for bidirectional ports
- Other special types (like `sim::DynamicStringType`) may need special handling
- Current code doesn't check if the converted type is actually hardware-compatible

**Mechanism**:
The `getModulePortInfo` function should validate that converted types are hardware-compatible before constructing `hw::PortInfo`.

## Suggested Fix Directions

1. **Add early validation in Moore dialect** (Recommended)
   - Check port types during Moore IR construction
   - Emit diagnostic for non-synthesizable types used as ports
   - Location: Moore dialect parsing/validation

2. **Add type validation in `getModulePortInfo()`**
   - Check if `convertType()` result is a valid hardware type
   - Use `hw::isHWValueType()` or similar to validate
   - Emit proper error instead of assertion failure

3. **Return `std::nullopt` from type converter**
   - Modify `StringType` conversion to return `std::nullopt` when used in port context
   - Or add a different conversion path for port types vs internal types

4. **Improve error message**
   - At minimum, the assertion should be replaced with a proper diagnostic
   - Message: "string type cannot be used as a module port - not synthesizable"

## Keywords for Issue Search
`string` `DynamicStringType` `module port` `MooreToCore` `getModulePortInfo` `hw::PortInfo` `unsynthesizable`

## Related Files
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Conversion pass implementation
- `include/circt/Dialect/HW/HWTypes.h` - HW type definitions
- `include/circt/Dialect/Moore/MooreTypes.td` - Moore type definitions
- `include/circt/Dialect/Sim/SimTypes.td` - Sim dialect types (DynamicStringType)
