# Root Cause Analysis Report

## Executive Summary

The crash occurs in MooreToCore pass when converting a SystemVerilog module with `string` type ports. The type converter successfully converts Moore's `StringType` to `sim::DynamicStringType`, but the HW dialect's module port infrastructure does not support this type. When `getModulePortInfo()` constructs `hw::PortInfo` with the converted type, a subsequent `dyn_cast<InOutType>` operation fails because `DynamicStringType` is not a valid HW port type, causing the assertion failure.

## Crash Context

- **Tool**: circt-verilog
- **Dialect**: Moore (SystemVerilog frontend)
- **Failing Pass**: MooreToCorePass
- **Crash Type**: Assertion failure
- **CIRCT Version**: 1.139.0

## Error Analysis

### Assertion Message
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Key Stack Frames
```
#11 getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp)
    MooreToCore.cpp:259
#12 SVModuleOpConversion::matchAndRewrite(...)
    MooreToCore.cpp:276
#34 MooreToCorePass::runOnOperation()
    MooreToCore.cpp:2571
```

### Crash Location
- **File**: `lib/Conversion/MooreToCore/MooreToCore.cpp`
- **Line**: 259 (in `getModulePortInfo()` function)
- **Function**: `getModulePortInfo(const TypeConverter &typeConverter, SVModuleOp op)`

## Test Case Analysis

### Code Summary
```systemverilog
module string_register(
  input logic clk,
  input string data_in,    // String type as input port
  output string data_out   // String type as output port
);
  string a = "Test";
  assign data_in = a;      // Assigning to input port (also problematic)
  always @(posedge clk) begin
    data_out <= data_in;
  end
endmodule
```

### Key Constructs
1. **`string` type ports** - SystemVerilog `string` type used as module port types
2. **`assign` to input port** - `assign data_in = a;` assigns to an input (unusual but not directly causing crash)
3. **Sequential logic with string** - `always @(posedge clk)` block operating on strings

### Problematic Patterns
The primary issue is using `string` type for module ports (`input string data_in`, `output string data_out`). This is valid SystemVerilog but not supported in CIRCT's hardware synthesis flow.

## CIRCT Source Analysis

### Crash Location Code
```cpp
// MooreToCore.cpp:234-259
static hw::ModulePortInfo getModulePortInfo(const TypeConverter &typeConverter,
                                            SVModuleOp op) {
  size_t inputNum = 0;
  size_t resultNum = 0;
  auto moduleTy = op.getModuleType();
  SmallVector<hw::PortInfo> ports;
  ports.reserve(moduleTy.getNumPorts());

  for (auto port : moduleTy.getPorts()) {
    Type portTy = typeConverter.convertType(port.type);  // Line 243
    // ... construct hw::PortInfo with portTy
  }
  return hw::ModulePortInfo(ports);  // Line 259
}
```

### Type Conversion Logic
```cpp
// MooreToCore.cpp:2277-2279
typeConverter.addConversion([&](StringType type) {
  return sim::DynamicStringType::get(type.getContext());
});
```

### Processing Path
1. Parse SystemVerilog `string` type port declarations
2. Create Moore dialect representation with `StringType`
3. MooreToCorePass attempts conversion
4. `getModulePortInfo()` calls `typeConverter.convertType(port.type)`
5. `StringType` → `sim::DynamicStringType` conversion succeeds
6. `hw::PortInfo` is created with `DynamicStringType`
7. **[FAILS]** Later code attempts `dyn_cast<InOutType>` on a type that is not supported

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence) ⭐
**Cause**: HW dialect does not support `sim::DynamicStringType` as a valid module port type

**Evidence**:
- Type conversion exists: `StringType` → `sim::DynamicStringType`
- `DynamicStringType` is in `sim` dialect, not `hw` dialect
- HW module ports expect hardware-synthesizable types (integers, arrays, structs)
- No validation exists to reject unsupported port types before construction

**Mechanism**: 
The type converter happily converts `StringType` to `DynamicStringType`, but this type cannot be used in HW module ports. When `hw::ModulePortInfo` is constructed and subsequently processed, internal operations expecting hardware types (like `dyn_cast<InOutType>`) fail because `DynamicStringType` is not a hardware type.

### Hypothesis 2 (Medium Confidence)
**Cause**: Missing type conversion validation for port types

**Evidence**:
- No check for `if (!portTy)` after `convertType()` in `getModulePortInfo()`
- Other conversion functions have such validation (e.g., line 789: "failed to convert result type")
- The conversion succeeds but produces an incompatible type

**Mechanism**:
The code assumes that if `convertType()` returns a non-null type, it's valid for the target context. However, `DynamicStringType` is valid as a type but invalid as a port type.

### Hypothesis 3 (Lower Confidence)
**Cause**: `string` type ports should be rejected earlier in the pipeline

**Evidence**:
- String is a simulation-only type in SystemVerilog
- Cannot be synthesized to hardware
- Should be caught during elaboration/import, not during lowering

## Suggested Fix Directions

### Option 1: Early Rejection (Recommended)
Add validation in `getModulePortInfo()` to check if converted port types are HW-compatible:
```cpp
Type portTy = typeConverter.convertType(port.type);
if (!portTy || !isHWCompatibleType(portTy)) {
  return op.emitError("unsupported port type: ") << port.type;
}
```

### Option 2: Type Converter Guard
Make the `StringType` conversion conditional on context:
```cpp
typeConverter.addConversion([&](StringType type) -> std::optional<Type> {
  // Only convert in non-port contexts
  return sim::DynamicStringType::get(type.getContext());
});
```
And add a target materialization that fails for port contexts.

### Option 3: Slang-Level Rejection
Reject `string` type ports during SystemVerilog import in the Slang frontend.

## Keywords for Issue Search
`string` `port` `DynamicStringType` `MooreToCore` `getModulePortInfo` `InOutType` `dyn_cast`

## Related Files
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Crash location
- `include/circt/Dialect/Moore/MooreTypes.td` - Moore dialect type definitions
- `include/circt/Dialect/Sim/SimTypes.td` - Sim dialect type definitions (DynamicStringType)
- `include/circt/Dialect/HW/HWTypes.td` - HW dialect type definitions

## Impact Assessment
- **Severity**: Medium (crash, but on invalid input)
- **Scope**: Affects any SystemVerilog code using `string` type ports
- **Workaround**: Avoid using `string` type for module ports
