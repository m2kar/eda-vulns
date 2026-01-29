# Root Cause Analysis Report

## Executive Summary
The crash occurs when `circt-verilog` attempts to convert a SystemVerilog module with a `string` type output port from the Moore dialect to the HW dialect. The type conversion produces `sim::DynamicStringType`, which is not a valid hardware port type, leading to an assertion failure when constructing the `hw::PortInfo` structure.

## Crash Context
- **Tool**: circt-verilog --ir-hw
- **Version**: CIRCT firtool-1.139.0, LLVM 22.0.0git
- **Dialect**: Moore
- **Failing Pass**: MooreToCore conversion
- **Crash Type**: Segmentation fault (originally assertion failure)

## Error Analysis
### Assertion Message (Original)
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Key Stack Frames
```
#4 SVModuleOpConversion::matchAndRewrite() MooreToCore.cpp
#5 ConversionPattern::dispatchTo1To1<SVModuleOp>
#16 MooreToCorePass::runOnOperation()
```

## Test Case Analysis
### Code Summary
```systemverilog
module test(input signed [5:0] a, output signed [3:0] b, output string msg);
  string str_var = "Test";
  assign b = a / 4'sd2;
  assign msg = str_var;
  initial begin
    $display("b=%0d, msg=%s", b, msg);
  end
endmodule
```

### Key Constructs
1. Module with `output string msg` - string type used as output port
2. String variable `str_var` initialized with literal
3. Continuous assignment of string to output port

### Problematic Pattern
**String type as module port**: SystemVerilog allows `string` type as a port, but this is not synthesizable hardware and the Moore-to-HW conversion cannot handle it.

## CIRCT Source Analysis

### Crash Location
**File**: `lib/Conversion/MooreToCore/MooreToCore.cpp`
**Function**: `getModulePortInfo()` (line 234-259)

### Code Context
```cpp
static hw::ModulePortInfo getModulePortInfo(const TypeConverter &typeConverter,
                                            SVModuleOp op) {
  // ...
  for (auto port : moduleTy.getPorts()) {
    Type portTy = typeConverter.convertType(port.type);  // <-- Returns sim::DynamicStringType
    if (port.dir == hw::ModulePort::Direction::Output) {
      ports.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, resultNum++, {}}));  // <-- CRASH HERE
    }
    // ...
  }
  return hw::ModulePortInfo(ports);
}
```

### Type Conversion Chain
1. Moore `StringType` → `sim::DynamicStringType` (line 2277-2279)
2. `sim::DynamicStringType` is **NOT** a valid `hw::` type for ports
3. `hw::PortInfo` constructor cannot handle non-hardware types

### Processing Path
1. Parse SystemVerilog → Moore dialect (success)
2. Moore module with string port created
3. MooreToCore pass attempts conversion
4. `getModulePortInfo()` converts port types
5. String port → `sim::DynamicStringType` (success)
6. Create `hw::PortInfo` with non-hw type → **CRASH**

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence) ⭐
**Cause**: `sim::DynamicStringType` is not a valid hardware port type, and the conversion does not validate or reject unsupported types before constructing `hw::PortInfo`.

**Evidence**:
- Type converter exists for `StringType` → `sim::DynamicStringType` (line 2277)
- `getModulePortInfo()` blindly uses converted type without validation
- HW dialect ports require types like `IntegerType`, `hw::ArrayType`, `hw::StructType`
- No error handling for unsupported port types

**Mechanism**:
The conversion infrastructure assumes all converted types are valid for hardware modules. When a `string` port is encountered, its type converts successfully to `sim::DynamicStringType`, but this type cannot be used in `hw::PortInfo`. The construction of `hw::ModulePortInfo` then fails.

### Hypothesis 2 (Medium Confidence)
**Cause**: Missing type conversion rejection for simulation-only types in port contexts.

**Evidence**:
- Other non-synthesizable types (like `chandle`, `class`) return `std::optional<Type>{}` to signal conversion failure
- String conversion unconditionally returns `sim::DynamicStringType`
- Should return `std::nullopt` when used in synthesizable context (module ports)

## Suggested Fix Directions

1. **Add validation in `getModulePortInfo()`**:
   ```cpp
   Type portTy = typeConverter.convertType(port.type);
   if (!portTy || !hw::isHWType(portTy)) {
     // Emit error: unsupported port type
     return failure();
   }
   ```

2. **Improve type converter to context-aware rejection**:
   - String type should fail conversion when target is hardware module port
   - Similar to how `ChandleType` conversion returns `std::nullopt`

3. **Alternative: Emit proper diagnostic instead of crash**:
   ```cpp
   if (auto strType = dyn_cast<sim::DynamicStringType>(portTy)) {
     op.emitError("string type ports are not supported in hardware synthesis");
     return failure();
   }
   ```

## Keywords for Issue Search
`string` `port` `MooreToCore` `DynamicStringType` `getModulePortInfo` `hw::PortInfo` `type conversion`

## Related Files
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Main conversion logic
- `include/circt/Dialect/Moore/MooreTypes.h` - Moore StringType definition
- `include/circt/Dialect/Sim/SimTypes.h` - Sim DynamicStringType definition
- `include/circt/Dialect/HW/HWTypes.h` - HW type definitions

## Severity Assessment
- **Impact**: Crash (denial of service for valid SystemVerilog input)
- **Likelihood**: Low (string ports are uncommon in synthesizable code)
- **Priority**: Medium - Should emit proper error instead of crashing
