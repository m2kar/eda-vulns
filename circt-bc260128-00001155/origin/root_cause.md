# Root Cause Analysis Report

## Executive Summary

circt-verilog crashes with assertion failure "dyn_cast on a non-existent value" when processing a SystemVerilog module with `string` type port. The root cause is that **`moore::StringType` lacks a type conversion rule in MooreToCore pass**, causing the type converter to return a null/invalid type that triggers the assertion in `sanitizeInOut()`.

## Crash Context

- **Tool/Command**: `circt-verilog --ir-hw source.sv`
- **Dialect**: Moore (circt-verilog frontend) → HW conversion
- **Failing Pass**: MooreToCore (SVModuleOpConversion)
- **Crash Type**: Assertion Failure
- **Exit Code**: 139 (SIGABRT)

## Error Analysis

### Assertion/Error Message
```
circt-verilog: .../llvm/include/llvm/Support/Casting.h:650: 
decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]: 
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Key Stack Frames
```
#17 circt::hw::ModulePortInfo::sanitizeInOut() PortImplementation.h:177:24
#21 (anonymous namespace)::getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp) MooreToCore.cpp:259:1
#22 (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(...) MooreToCore.cpp:276:32
#42 (anonymous namespace)::MooreToCorePass::runOnOperation() MooreToCore.cpp:2571:14
```

## Test Case Analysis

### Code Summary
```systemverilog
module string_processor(input logic clk, input string a);
  string data_array [0:3];
  int result;
  
  always @(posedge clk) begin
    data_array[0] = a.toupper();
    result = data_array[0].len();
  end
endmodule
```

The test case uses SystemVerilog `string` type:
1. As a **module port** (`input string a`)
2. As an **array element type** (`string data_array [0:3]`)
3. With **string built-in methods** (`.toupper()`, `.len()`)

### Key Constructs
- `input string a` - **Module port with string type** (triggers the crash)
- `string data_array [0:3]` - Unpacked array of strings
- `.toupper()`, `.len()` - String built-in methods

### Potentially Problematic Pattern
**Using `string` type as module port** - The `string` type in Moore dialect has no corresponding type in HW dialect, and there's no conversion rule defined.

## CIRCT Source Analysis

### Crash Location
**File**: `include/circt/Dialect/HW/PortImplementation.h`
**Function**: `ModulePortInfo::sanitizeInOut()`
**Line**: 177

### Code Context
```cpp
// PortImplementation.h:175-181
private:
  void sanitizeInOut() {
    for (auto &p : ports)
      if (auto inout = dyn_cast<hw::InOutType>(p.type)) {  // <-- crash here
        p.type = inout.getElementType();
        p.dir = ModulePort::Direction::InOut;
      }
  }
```

The `dyn_cast` fails because `p.type` is null/invalid (not a valid MLIR Type).

### Processing Path

1. **SVModuleOpConversion::matchAndRewrite()** (MooreToCore.cpp:276)
   - Calls `getModulePortInfo(*typeConverter, op)`

2. **getModulePortInfo()** (MooreToCore.cpp:241-258)
   ```cpp
   for (auto port : moduleTy.getPorts()) {
     Type portTy = typeConverter.convertType(port.type);  // <-- returns null for StringType
     // ... creates PortInfo with null type
   }
   return hw::ModulePortInfo(ports);  // <-- triggers sanitizeInOut()
   ```

3. **ModulePortInfo constructor** calls `sanitizeInOut()` which crashes on null type

### Missing Type Conversion

In **MooreToCore.cpp:2220-2362** (`populateTypeConversion()`), the following types have conversion rules:
- `IntType` → `IntegerType`
- `RealType` → `Float32Type/Float64Type`
- `TimeType` → `llhd::TimeType`
- `FormatStringType` → `sim::FormatStringType`
- `ArrayType` → `hw::ArrayType`
- `StructType` → `hw::StructType`
- `ChandleType` → `LLVM::LLVMPointerType`
- `ClassHandleType` → `LLVM::LLVMPointerType`
- `RefType` → `llhd::RefType`

**`moore::StringType` is NOT in this list!**

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: Missing type conversion rule for `moore::StringType` in `populateTypeConversion()`

**Evidence**:
1. `StringType` is defined in `MooreTypes.td` (line 40) as a valid Moore type
2. `FormatStringType` has conversion to `sim::FormatStringType`, but `StringType` does not
3. The crash occurs during type conversion when processing module ports
4. No `typeConverter.addConversion([&](StringType type)...)` exists in the codebase

**Mechanism**: 
1. Module port has `StringType`
2. `typeConverter.convertType()` returns null (no conversion rule)
3. Null type stored in `PortInfo`
4. `dyn_cast` on null type triggers assertion

### Hypothesis 2 (Medium Confidence)
**Cause**: Missing error handling for unconvertible types in `getModulePortInfo()`

**Evidence**:
1. `getModulePortInfo()` doesn't check if `convertType()` succeeded
2. Other conversion patterns (e.g., `ArrayType`) check for conversion failure and return `{}`
3. Proper error handling would have emitted a diagnostic instead of crashing

**Mechanism**: Even if `StringType` isn't supported, the code should gracefully fail with an error message instead of crashing.

## Suggested Fix Directions

1. **Add StringType conversion** (if string support is desired):
   ```cpp
   // In populateTypeConversion():
   typeConverter.addConversion([&](StringType type) -> std::optional<Type> {
     // Option A: Map to a pointer type (for simulation)
     return LLVM::LLVMPointerType::get(type.getContext());
     // Option B: Return failure to emit proper error
     // return {};
   });
   ```

2. **Add null check in getModulePortInfo()** (defensive fix):
   ```cpp
   Type portTy = typeConverter.convertType(port.type);
   if (!portTy) {
     // Emit error about unsupported port type
     return failure();
   }
   ```

3. **Document unsupported types** - If `string` ports aren't supported, emit a clear diagnostic.

## Keywords for Issue Search

`StringType` `MooreToCore` `sanitizeInOut` `dyn_cast non-existent` `type conversion` `module port` `string port` `circt-verilog crash`

## Related Files to Investigate

- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Add StringType conversion
- `include/circt/Dialect/HW/PortImplementation.h` - Add defensive null check
- `include/circt/Dialect/Moore/MooreTypes.td` - StringType definition
- `include/circt/Dialect/Moore/MooreOps.td` - Operations using StringType
