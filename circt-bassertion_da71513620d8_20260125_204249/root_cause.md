# Root Cause Analysis Report

## Executive Summary

CIRCT crashes when processing a SystemVerilog module with a `string` type as an output port. The crash occurs in the MooreToCore conversion pass when attempting to create an HW module with a port of type `sim::DynamicStringType`, which is a simulation type incompatible with hardware module ports.

## Crash Context

- **Tool/Command**: `circt-verilog --ir-hw source.sv`
- **Dialect**: Moore (SystemVerilog frontend)
- **Failing Pass**: MooreToCore (converting Moore dialect to HW/Core dialects)
- **Crash Type**: Segmentation fault / Assertion failure

## Error Analysis

### Assertion/Error Message

**Original error log**:
```
circt-verilog: /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:650: decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]: Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

**Reproduce output**:
- Exit code: 139 (SIGSEGV)
- Stack trace shows crash at `SVModuleOpConversion::matchAndRewrite`

### Key Stack Frames

```
#4  SVModuleOpConversion::matchAndRewrite(...) at MooreToCore.cpp:0
#276  getModulePortInfo(*typeConverter, op) called in SVModuleOpConversion
#259  getModulePortInfo() returns hw::ModulePortInfo
```

The crash occurs when `SVModuleOpConversion::matchAndRewrite` calls `getModulePortInfo` to extract port information for creating an `hw::ModuleOp`.

## Test Case Analysis

### Code Summary

The test case defines a SystemVerilog module with a `string` type output port:

```systemverilog
module test_module(
  input logic clk,
  input logic [31:0] data_in,
  output string str_out    // <-- PROBLEMATIC: string type as output port
);
  // ... logic ...
endmodule
```

### Key Constructs

- `output string str_out` - String type used as a module output port
- `always_ff`, `always_comb` - Standard procedural blocks
- `assign str_out = str` - String assignment

### Potentially Problematic Patterns

**Primary issue**: Using `string` type as a module port is problematic because:
1. `string` is a high-level SystemVerilog construct primarily for simulation
2. Hardware synthesis typically requires fixed-width types
3. CIRCT's Moore dialect converts `string` to `sim::DynamicStringType`
4. Simulation types are not valid for HW module ports

## CIRCT Source Analysis

### Crash Location

**File**: `lib/Conversion/MooreToCore/MooreToCore.cpp`
**Function**: `SVModuleOpConversion::matchAndRewrite` (line 268-291)
**Call chain**: `SVModuleOpConversion::matchAndRewrite` → `getModulePortInfo` (line 276)

### Code Context

**TypeConverter for StringType** (MooreToCore.cpp:2277-2279):
```cpp
typeConverter.addConversion([&](StringType type) {
    return sim::DynamicStringType::get(type.getContext());
});
```

**Port extraction** (MooreToCore.cpp:242-255):
```cpp
for (auto port : moduleTy.getPorts()) {
    Type portTy = typeConverter.convertType(port.type);  // Line 243
    if (port.dir == hw::ModulePort::Direction::Output) {
      ports.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, resultNum++, {}}));
    } else {
      ports.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, inputNum++, {}}));
    }
  }
```

### Processing Path

1. **Parse SystemVerilog**: Moore dialect frontend parses the module
2. **Type conversion**: `StringType` port is converted to `sim::DynamicStringType`
3. **Module creation**: `SVModuleOpConversion::matchAndRewrite` calls `getModulePortInfo`
4. **Port extraction**: `getModulePortInfo` creates `hw::PortInfo` with `sim::DynamicStringType` as port type
5. **HW module creation**: Pass attempts to create `hw::ModuleOp` with simulation-type port
6. **CRASH**: Downstream code cannot handle `sim::DynamicStringType` in hardware context
   - Likely when trying to `dyn_cast` to `InOutType` or other hardware types
   - Or when processing ports in HW dialect operations

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)

**Cause**: CIRCT does not support `string` type as module ports for synthesis to HW dialect, but lacks proper validation/error handling.

**Evidence**:
- Test case has `output string str_out` which is a SystemVerilog string type
- TypeConverter converts `StringType` → `sim::DynamicStringType` (line 2277-2279)
- `sim::DynamicStringType` is a simulation type defined in the Sim dialect
- HW module ports expect hardware types (IntegerType, hw::ArrayType, etc.), not simulation types
- Crash occurs when creating `hw::ModuleOp` with incompatible port type

**Mechanism**:
The MooreToCore pass attempts to convert a Moore `SVModuleOp` (with a `string` port) to an HW `ModuleOp`. The TypeConverter correctly converts the string type to `sim::DynamicStringType` for simulation purposes. However, when creating the hardware module, the port type is still `sim::DynamicStringType`, which is incompatible with hardware module port expectations. Downstream code attempting to process this port (e.g., casting to `InOutType` or other HW types) fails because the type is invalid in the hardware context.

### Hypothesis 2 (Medium Confidence)

**Cause**: `sim::DynamicStringType` may be missing proper support in HW dialect's type system.

**Evidence**:
- Sim dialect types are generally used for simulation constructs, not synthesis
- HW dialect may not have handlers for Sim dialect types
- No explicit validation or error when passing Sim types to HW modules

**Mechanism**:
The HW dialect's infrastructure (ModulePortInfo, PortInfo, etc.) expects only certain types (IntegerType, ArrayType, StructType, etc.). When a `sim::DynamicStringType` is passed, the infrastructure may lack proper handling, leading to undefined behavior or crashes when attempting to access type-specific properties.

### Hypothesis 3 (Low Confidence)

**Cause**: Type conversion pipeline may have a bug in handling mixed dialect types.

**Evidence**:
- Mixed dialect scenario: Moore → Sim → HW
- TypeConverter correctly converts StringType → DynamicStringType
- Issue occurs in module construction, not type conversion itself

**Mechanism**:
The conversion pipeline may not properly validate that all types are legal for the target dialect (HW) before creating operations. A validation step might be missing that should catch incompatible types like `sim::DynamicStringType` in HW modules.

## Suggested Fix Directions

1. **Add validation in TypeConverter** (Recommended):
   - In `populateTypeConversion`, add a check that prevents conversion of StringType when used as a module port
   - Return a proper error message: "String type is not supported as a module port for hardware synthesis"

2. **Add validation in `getModulePortInfo`**:
   - Check if `portTy` is a valid hardware type before adding to `ports`
   - Reject `sim::DynamicStringType` and other simulation types with a clear error

3. **Mark simulation types as illegal for HW modules**:
   - In the `MooreToCore` pass's conversion target, mark operations with `sim::DynamicStringType` ports as illegal
   - This will cause the conversion framework to report an error rather than crash

4. **Improve error handling** (Short-term fix):
   - Add assertions with better error messages in `getModulePortInfo`
   - Check for invalid types before creating `hw::PortInfo`
   - At minimum, change the crash to a proper diagnostic error

## Keywords for Issue Search

`string port`, `DynamicStringType`, `MooreToCore`, `module port`, `simulation type`, `HW dialect`, `type conversion`, `SystemVerilog string`, `port type`, `crash`

## Related Files to Investigate

- `lib/Conversion/MooreToCore/MooreToCore.cpp` - TypeConverter and module conversion
- `include/circt/Dialect/Sim/SimTypes.td` - DynamicStringType definition
- `include/circt/Dialect/HW/HWTypes.h` - ModulePort and HW type definitions
- `include/circt/Dialect/HW/PortImplementation.h` - PortInfo structure
- `lib/Dialect/Moore/` - Moore dialect type definitions and operations

## Additional Notes

- The crash signature (`dyn_cast on a non-existent value`) suggests that the `sim::DynamicStringType` object may be null or invalid in certain code paths
- The issue is specific to using `string` as a **module port**; string variables inside modules may work differently
- This is likely an unsupported feature that should be explicitly rejected rather than causing a crash
