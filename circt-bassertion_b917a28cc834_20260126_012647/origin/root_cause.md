# Root Cause Analysis Report

## Executive Summary

The CIRCT compiler crashes when processing a SystemVerilog module with a `string` type input port. The crash occurs in the MooreToCore conversion pass when `getModulePortInfo()` attempts to convert port types. The type converter returns `nullptr` for `StringType` (in CIRCT version 1.139.0), and this null type is later passed to `hw::ModulePortInfo` constructor which calls `dyn_cast<hw::InOutType>` on it, triggering an assertion failure "dyn_cast on a non-existent value".

## Crash Context

- **Tool**: circt-verilog
- **Dialect**: Moore
- **Failing Pass**: MooreToCore (ConvertMooreToCore)
- **Crash Type**: Assertion failure
- **CIRCT Version**: 1.139.0

## Error Analysis

### Assertion Message
```
circt-verilog: .../llvm/llvm/include/llvm/Support/Casting.h:650: 
decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]: 
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Key Stack Frames
```
#11 0x000055e7590cab57 (assertion triggered in dyn_cast)
#12 0x000055e7596d3717 llvm::SmallVector<circt::hw::PortInfo, 1u>::~SmallVector()
#13 0x000055e7596d3717 (anonymous namespace)::getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp) MooreToCore.cpp:259
#14 0x000055e7596d3717 (anonymous namespace)::SVModuleOpConversion::matchAndRewrite MooreToCore.cpp:276
#35 0x000055e75965fb00 (anonymous namespace)::MooreToCorePass::runOnOperation() MooreToCore.cpp:2571
```

### Crash Site Analysis
The crash occurs in `ModulePortInfo::sanitizeInOut()` (called from `ModulePortInfo` constructor), which iterates over ports and calls `dyn_cast<hw::InOutType>(p.type)`. When `p.type` is null, the assertion fails.

## Test Case Analysis

### Code Summary
```systemverilog
module test_module(input logic clk, input string a);
  logic r1;
  int b;
  
  always @(posedge clk) begin
    b = a.len();
    r1 = (b > 0) ? 1'b1 : 1'b0;
  end
endmodule
```

The test declares a module with:
1. A clock input (`logic` type)
2. A string input (`string` type)
3. An always block that calls `.len()` method on the string

### Key Constructs
- `input string a` - String type as module input port
- `a.len()` - String method invocation
- `always @(posedge clk)` - Procedural block

### Problematic Patterns
The core issue is `input string a` - using SystemVerilog `string` type as a module port. The `string` type is a dynamic, variable-length data type in SystemVerilog that doesn't map naturally to hardware synthesis semantics.

## CIRCT Source Analysis

### Crash Location
- **File**: `lib/Conversion/MooreToCore/MooreToCore.cpp:259`
- **Function**: `getModulePortInfo(const TypeConverter &typeConverter, SVModuleOp op)`

### Code Context (from GitHub master)
```cpp
/// Get the ModulePortInfo from a SVModuleOp.
static hw::ModulePortInfo getModulePortInfo(const TypeConverter &typeConverter,
                                            SVModuleOp op) {
  size_t inputNum = 0;
  size_t resultNum = 0;
  auto moduleTy = op.getModuleType();
  SmallVector<hw::PortInfo> ports;
  ports.reserve(moduleTy.getNumPorts());

  for (auto port : moduleTy.getPorts()) {
    Type portTy = typeConverter.convertType(port.type);  // <-- Returns nullptr for StringType
    if (port.dir == hw::ModulePort::Direction::Output) {
      ports.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, resultNum++, {}}));
    } else {
      ports.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, inputNum++, {}}));
    }
  }

  return hw::ModulePortInfo(ports);  // <-- sanitizeInOut() called here, crashes on null type
}
```

### Type Conversion Analysis
In the current master branch, `StringType` is converted to `sim::DynamicStringType`:
```cpp
typeConverter.addConversion([&](StringType type) {
  return sim::DynamicStringType::get(type.getContext());
});
```

However, in CIRCT 1.139.0, this conversion may not exist or may be incomplete, causing `typeConverter.convertType(port.type)` to return `nullptr` for `StringType`.

### Processing Path
1. **ImportVerilog**: Parses SystemVerilog, creates `moore::StringType` for `input string a`
2. **Moore Dialect**: Module has port with `moore::StringType`
3. **MooreToCore Pass**: Attempts to convert module ports to HW dialect
4. **getModulePortInfo()**: Calls `typeConverter.convertType(port.type)` for each port
5. **Type Conversion**: Returns `nullptr` for `StringType` (no conversion registered in v1.139.0)
6. **ModulePortInfo Constructor**: Calls `sanitizeInOut()` which performs `dyn_cast<InOutType>` on null type
7. **CRASH**: Assertion failure in `dyn_cast` because type is null

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: Missing or incomplete type conversion for `moore::StringType` in MooreToCore pass (CIRCT v1.139.0)

**Evidence**:
- Test uses `input string a` which creates `moore::StringType` for the port
- `typeConverter.convertType()` returns `nullptr` when no conversion is registered
- The null type is stored in `hw::PortInfo` and passed to `ModulePortInfo` constructor
- `sanitizeInOut()` crashes when attempting `dyn_cast<InOutType>` on null type

**Mechanism**: 
The type converter in CIRCT 1.139.0 does not have a registered conversion for `moore::StringType`. When `convertType()` is called on a `StringType`, it returns `nullptr`. This null value propagates through the port construction and eventually causes the crash in `sanitizeInOut()`.

### Hypothesis 2 (Medium Confidence)
**Cause**: `getModulePortInfo()` does not validate the result of type conversion

**Evidence**:
- The code does not check if `portTy` is null after `typeConverter.convertType()`
- Any unsupported type would cause the same crash pattern
- This is a missing defensive check rather than a fundamental design issue

**Mechanism**:
Even if `StringType` conversion is added, any other unconvertible type would trigger the same crash. The function should either validate the conversion result or emit a proper error diagnostic.

### Hypothesis 3 (Lower Confidence)
**Cause**: SystemVerilog `string` type is not synthesizable and should be rejected earlier

**Evidence**:
- `string` is a dynamic type not suitable for hardware synthesis
- Other tools (like Verilator) may reject string ports
- CIRCT should emit a meaningful error rather than crash

**Mechanism**:
The proper fix might be to reject `string` type in module ports during an earlier validation pass, providing a clear error message that string types are not supported in synthesizable contexts.

## Suggested Fix Directions

1. **Add Type Conversion (Preferred)**:
   - Register conversion for `moore::StringType` → `sim::DynamicStringType` (or appropriate target type)
   - This appears to be done in the master branch but may not be in v1.139.0

2. **Add Null Check in getModulePortInfo()**:
   ```cpp
   for (auto port : moduleTy.getPorts()) {
     Type portTy = typeConverter.convertType(port.type);
     if (!portTy) {
       // Emit error: unsupported port type
       return failure();  // or return empty ModulePortInfo
     }
     // ... rest of the code
   }
   ```

3. **Reject Unsupported Types Earlier**:
   - Add validation in `SVModuleOp` verification
   - Emit diagnostic: "string type is not supported in module ports"

4. **Fix sanitizeInOut()** (Defensive):
   - Add null check before `dyn_cast`:
   ```cpp
   if (p.type && isa<hw::InOutType>(p.type)) {
     // ...
   }
   ```

## Keywords for Issue Search
`string` `module port` `InOutType` `MooreToCore` `dyn_cast` `StringType` `type conversion` `getModulePortInfo` `sanitizeInOut`

## Related Files
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Crash site and type conversion
- `include/circt/Dialect/HW/PortImplementation.h` - `ModulePortInfo::sanitizeInOut()`
- `lib/Conversion/ImportVerilog/Types.cpp` - `moore::StringType` creation
- `include/circt/Dialect/Moore/MooreTypes.td` - Moore dialect type definitions
- `include/circt/Dialect/Sim/SimTypes.td` - Target type `DynamicStringType`

## Additional Notes

The bug appears to be fixed in the master branch where `StringType` conversion is properly registered. This suggests this is a version-specific issue in CIRCT 1.139.0. Users experiencing this crash should:
1. Update to a newer version of CIRCT if available
2. Avoid using `string` type in module ports as a workaround
3. Use non-synthesizable testbench code separately from synthesizable modules
