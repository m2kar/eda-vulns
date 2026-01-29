# Root Cause Analysis Report

## Executive Summary

`circt-verilog` crashes with an assertion failure when processing a SystemVerilog module that has a `string` type on a **module port**. The `string` type is correctly converted to `sim::DynamicStringType`, but this type is not a valid hardware type for module ports in the HW dialect. When the `ModulePortInfo` constructor calls `sanitizeInOut()`, it attempts to `dyn_cast<hw::InOutType>` on the port type, which fails the `detail::isPresent(Val)` assertion because `sim::DynamicStringType` is not compatible with HW port types.

## Crash Context
- **Tool/Command**: `circt-verilog --ir-hw`
- **Dialect**: Moore → HW conversion
- **Failing Pass**: `MooreToCore` (specifically `SVModuleOpConversion`)
- **Crash Type**: Assertion failure

## Error Analysis

### Assertion/Error Message
```
circt-verilog: .../llvm/include/llvm/Support/Casting.h:650: 
decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]: 
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Key Stack Frames
```
#11 MooreToCore.cpp - in destructor ~SmallVector<hw::PortInfo>
#12 getModulePortInfo(TypeConverter const&, SVModuleOp) @ MooreToCore.cpp:259
#13 SVModuleOpConversion::matchAndRewrite() @ MooreToCore.cpp:276
#34 MooreToCorePass::runOnOperation() @ MooreToCore.cpp:2571
```

## Test Case Analysis

### Code Summary
```systemverilog
module test_module(input logic in, output string str_out);
  string str;
  logic x;
  
  always_comb begin
    str = "test";
    x = in;
  end
  
  assign str_out = str;
endmodule
```

The test case declares a module with a `string` type **output port** (`str_out`), an internal `string` variable, and an `always_comb` block that assigns values.

### Key Constructs
- **`output string str_out`**: A string-typed output port (the triggering construct)
- **`string str`**: Internal string variable
- **`always_comb` block**: Procedural block with string assignment

### Potentially Problematic Patterns
- Using `string` type as a module **port** type is unusual in hardware synthesis
- `string` is a dynamic type that doesn't map naturally to fixed-width hardware types

## CIRCT Source Analysis

### Crash Location
**File**: `lib/Conversion/MooreToCore/MooreToCore.cpp`
**Function**: `getModulePortInfo()`
**Line**: 259 (return statement, where `ModulePortInfo` is constructed)

### Code Context

**getModulePortInfo (MooreToCore.cpp:234-259)**:
```cpp
static hw::ModulePortInfo getModulePortInfo(const TypeConverter &typeConverter,
                                            SVModuleOp op) {
  size_t inputNum = 0;
  size_t resultNum = 0;
  auto moduleTy = op.getModuleType();
  SmallVector<hw::PortInfo> ports;
  ports.reserve(moduleTy.getNumPorts());

  for (auto port : moduleTy.getPorts()) {
    Type portTy = typeConverter.convertType(port.type);  // <-- string → sim::DynamicStringType
    if (port.dir == hw::ModulePort::Direction::Output) {
      ports.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, resultNum++, {}}));
    } else {
      ports.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, inputNum++, {}}));
    }
  }

  return hw::ModulePortInfo(ports);  // <-- CRASH: constructor calls sanitizeInOut()
}
```

**ModulePortInfo constructor (PortImplementation.h:65-68)**:
```cpp
explicit ModulePortInfo(ArrayRef<PortInfo> mergedPorts)
    : ports(mergedPorts.begin(), mergedPorts.end()) {
  sanitizeInOut();  // <-- Calls problematic function
}
```

**sanitizeInOut (PortImplementation.h:175-181)**:
```cpp
void sanitizeInOut() {
  for (auto &p : ports)
    if (auto inout = dyn_cast<hw::InOutType>(p.type)) {  // <-- FAILS HERE
      p.type = inout.getElementType();
      p.dir = ModulePort::Direction::InOut;
    }
}
```

### Type Conversion Chain
1. **Source**: `string` (SystemVerilog string type)
2. **Moore dialect**: `moore::StringType`
3. **Conversion (MooreToCore.cpp:2277-2279)**:
   ```cpp
   typeConverter.addConversion([&](StringType type) {
     return sim::DynamicStringType::get(type.getContext());
   });
   ```
4. **Target**: `sim::DynamicStringType`

### Processing Path
1. `MooreToCorePass::runOnOperation()` starts the conversion
2. `SVModuleOpConversion::matchAndRewrite()` processes the module
3. `getModulePortInfo()` is called to convert module ports
4. For the `str_out` port:
   - `typeConverter.convertType(port.type)` converts `moore::StringType` → `sim::DynamicStringType`
   - A `hw::PortInfo` is created with this type
5. `hw::ModulePortInfo(ports)` constructor is called
6. `sanitizeInOut()` iterates over ports
7. `dyn_cast<hw::InOutType>(sim::DynamicStringType)` is called
8. **CRASH**: The dyn_cast fails because `sim::DynamicStringType` cannot be cast to `hw::InOutType`

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: `sim::DynamicStringType` is not a valid type for HW module ports, and the code doesn't validate or reject incompatible port types before creating `ModulePortInfo`.

**Evidence**:
- `sim::DynamicStringType` is a simulation-only type, not a synthesizable hardware type
- `sanitizeInOut()` assumes all port types are either:
  - `hw::InOutType` (which gets unwrapped), or
  - Some other valid HW type (which should pass through)
- There's no validation that the converted port type is actually a valid HW port type
- The assertion message confirms `dyn_cast` received an invalid/unexpected type

**Mechanism**: 
The type converter successfully converts `moore::StringType` to `sim::DynamicStringType`, but `sim::DynamicStringType` is not a hardware-representable type. The `hw::ModulePortInfo` infrastructure expects all ports to have types from the HW type system. When `sanitizeInOut()` runs, it tries to check if the type is `hw::InOutType` using `dyn_cast`, which fails on `sim::DynamicStringType`.

### Hypothesis 2 (Medium Confidence)
**Cause**: The `dyn_cast` implementation has a precondition that the source type must be in the "castable" type hierarchy, and `sim::DynamicStringType` is not in that hierarchy.

**Evidence**:
- The assertion is `detail::isPresent(Val)` - suggesting the type system doesn't recognize the type as valid for this cast
- `sim::DynamicStringType` is from the `sim` dialect, not the `hw` dialect
- The `dyn_cast` may require the type to at least be recognized as a potential `hw::Type`

**Mechanism**:
LLVM's `dyn_cast` implementation checks if the source type is even potentially compatible with the target type before attempting the cast. For types completely outside the HW type hierarchy, this check fails.

## Suggested Fix Directions

1. **Add port type validation in `getModulePortInfo()`**:
   ```cpp
   for (auto port : moduleTy.getPorts()) {
     Type portTy = typeConverter.convertType(port.type);
     if (!hw::isHWValueType(portTy)) {
       op.emitError() << "port '" << port.name << "' has unsupported type " << portTy;
       return {}; // or return failure
     }
     // ... rest of logic
   }
   ```

2. **Reject `string` type on ports earlier in the pipeline**:
   - Add a verification pass that checks for non-synthesizable types on ports
   - Emit a clear diagnostic instead of crashing

3. **Use `isa<>` instead of `dyn_cast<>` in `sanitizeInOut()`**:
   ```cpp
   void sanitizeInOut() {
     for (auto &p : ports)
       if (isa<hw::InOutType>(p.type))  // safer check
         if (auto inout = cast<hw::InOutType>(p.type)) {
           p.type = inout.getElementType();
           p.dir = ModulePort::Direction::InOut;
         }
   }
   ```
   However, this may mask the underlying issue of invalid port types.

4. **Document that `string` type ports are not supported**:
   - If this is intentionally unsupported, provide a clear error message

## Keywords for Issue Search
`string` `DynamicStringType` `ModulePortInfo` `InOutType` `dyn_cast` `MooreToCore` `port` `sanitizeInOut` `type conversion`

## Related Files to Investigate
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Main conversion logic
- `include/circt/Dialect/HW/PortImplementation.h` - `ModulePortInfo` and `sanitizeInOut()`
- `include/circt/Dialect/HW/HWTypes.h` - HW type checking utilities (`isHWValueType`)
- `include/circt/Dialect/Sim/SimTypes.td` - `DynamicStringType` definition
