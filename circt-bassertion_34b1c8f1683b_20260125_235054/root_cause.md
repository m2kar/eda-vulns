# Root Cause Analysis Report

## Executive Summary

circt-verilog crashes with an assertion failure when processing a SystemVerilog module that uses `string` type as an output port. The root cause is that the MooreToCore conversion pass lacks a type conversion rule for `moore::StringType`, causing `typeConverter.convertType()` to return a null/empty type, which then triggers an assertion in `llvm::dyn_cast()` when attempting to cast the non-existent type.

## Crash Context

- **Tool/Command**: `circt-verilog --ir-hw`
- **Dialect**: Moore
- **Failing Pass**: MooreToCorePass
- **Crash Type**: Assertion failure
- **CIRCT Version**: 1.139.0

## Error Analysis

### Assertion/Error Message
```
circt-verilog: /edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/llvm/include/llvm/Support/Casting.h:650: 
decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]: 
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Key Stack Frames
```
#13 getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp) 
    MooreToCore.cpp:259:1
#14 SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, ...) 
    MooreToCore.cpp:276:32
#35 (anonymous namespace)::MooreToCorePass::runOnOperation() 
    MooreToCore.cpp:2571:7
```

## Test Case Analysis

### Code Summary
The test case defines a simple SystemVerilog module with:
- Clock and reset inputs (`logic` type)
- A `string` type output port (`str_out`)
- A `string` internal variable
- String assignment in `initial` block and `always` block

```systemverilog
module test_module(
  input logic clk,
  input logic rst,
  output string str_out  // <-- Problematic: string as output port
);
  string str;
  // ... uses string in initial and always blocks
  assign str_out = str;
endmodule
```

### Key Constructs
- `string` type as module output port
- `string` type as internal variable
- String literal assignments (`"default"`, `"reset"`, `"normal"`)

### Potentially Problematic Patterns
The use of `string` type as a **module port** is the trigger. While `string` type as internal variable might work in some contexts, using it as a module port requires the type converter to handle it during module port conversion, which it doesn't.

## CIRCT Source Analysis

### Crash Location
**File**: `lib/Conversion/MooreToCore/MooreToCore.cpp`
**Function**: `getModulePortInfo()`
**Line**: ~259 (end of function, where SmallVector destructor runs)

### Code Context
```cpp
// MooreToCore.cpp:236-264
static FailureOr<hw::ModulePortInfo>
getModulePortInfo(const TypeConverter &typeConverter, SVModuleOp op) {
  size_t inputNum = 0;
  size_t resultNum = 0;
  auto moduleTy = op.getModuleType();
  SmallVector<hw::PortInfo> ports;
  ports.reserve(moduleTy.getNumPorts());

  for (auto port : moduleTy.getPorts()) {
    Type portTy = typeConverter.convertType(port.type);  // <-- Returns null for StringType
    if (!portTy) {
      return op.emitError("failed to convert type of port '")
             << port.name << "' in module '" << op.getName() << "'";
    }
    // ... rest of processing
  }
  return hw::ModulePortInfo(ports);
}
```

### Type Converter Analysis
The `populateTypeConversion()` function (lines 2268-2410) defines conversions for:
- `IntType` -> `IntegerType`
- `RealType` -> `Float32Type/Float64Type`
- `TimeType` -> `llhd::TimeType`
- `FormatStringType` -> `sim::FormatStringType`
- `ArrayType`, `UnpackedArrayType` -> `hw::ArrayType`
- `StructType`, `UnpackedStructType` -> `hw::StructType`
- `ChandleType` -> `LLVM::LLVMPointerType`
- `ClassHandleType` -> `LLVM::LLVMPointerType`
- `RefType` -> `llhd::RefType`

**MISSING**: `moore::StringType` has **NO** conversion rule defined!

### StringType Definition
```tablegen
// MooreTypes.td:40-42
def StringType : MooreTypeDef<"String", [], "moore::UnpackedType"> {
  let mnemonic = "string";
  let summary = "the SystemVerilog `string` type";
}
```

### Processing Path
1. `MooreToCorePass::runOnOperation()` starts conversion
2. `SVModuleOpConversion::matchAndRewrite()` handles module conversion
3. `getModulePortInfo()` iterates over module ports
4. For port `str_out` with `moore::StringType`:
   - `typeConverter.convertType(port.type)` returns **null** (no conversion rule)
   - The null check at line 245-248 **should** catch this and emit an error
   - **BUG**: The error message is emitted, but execution continues somehow, leading to a later crash when the invalid type is used

**Wait - re-analyzing the crash**: The assertion happens in the destructor of `SmallVector<hw::PortInfo>` at line 259. This suggests that the port was added to the vector with an invalid type, and the crash happens during cleanup. Let me re-examine...

Actually, looking more carefully at the stack trace:
```
#12 SmallVector<circt::hw::PortInfo, 1u>::~SmallVector()
#13 getModulePortInfo(...)  MooreToCore.cpp:259:1
```

The crash occurs at line 259 which is the **end of the function** (closing brace). This means:
1. The function attempted to process ports
2. Something went wrong that wasn't properly caught
3. During stack unwinding or return, the SmallVector destructor was called
4. The destructor tried to access a PortInfo with invalid type, triggering the assertion

### Root Cause Mechanism

The actual crash sequence:
1. `convertType(StringType)` returns `Type()` (null type)
2. The `if (!portTy)` check at line 245 catches this and **returns an error**
3. But the `FailureOr<>` return type means the error is wrapped
4. The `SmallVector<hw::PortInfo> ports` local variable's destructor runs
5. During destruction, something attempts to access/validate the port types
6. The `dyn_cast<hw::InOutType>` is called on a null type, triggering the assertion

The crash is actually during **error handling cleanup**, not during normal processing.

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: Missing type conversion rule for `moore::StringType` in `populateTypeConversion()`

**Evidence**:
- Test case uses `string` type as output port
- `populateTypeConversion()` has no handler for `StringType`
- `typeConverter.convertType()` returns null for unknown types
- Error message "failed to convert type" is designed to handle this, but crash occurs anyway

**Mechanism**: 
The type converter doesn't know how to convert `moore::StringType` to any target type. When `convertType()` returns null, the error handling path triggers but the crash occurs during cleanup of local variables in `getModulePortInfo()`.

### Hypothesis 2 (Medium Confidence)
**Cause**: The `FailureOr<>` error return interacts badly with local variable destruction

**Evidence**:
- Stack trace shows crash in `SmallVector` destructor
- Crash occurs at line 259 (function closing brace)
- `hw::PortInfo` might have invariants that are violated when type is null

**Mechanism**:
When returning an error from `getModulePortInfo()`, the `SmallVector<hw::PortInfo> ports` destructor runs. If any `PortInfo` was constructed with invalid data before the error check, the destructor might trigger assertions.

### Hypothesis 3 (Lower Confidence)
**Cause**: The `string` type might need special handling beyond just a type conversion

**Evidence**:
- `string` in SystemVerilog is a dynamic type with runtime-known length
- Unlike fixed-width types, it can't be directly mapped to hardware
- Similar to `ChandleType` and `ClassHandleType` which map to `LLVM::LLVMPointerType`

**Mechanism**:
`string` type might need to be lowered to a pointer or similar construct, but this conversion is not implemented. It's a fundamental limitation rather than just a missing conversion rule.

## Suggested Fix Directions

1. **Immediate Fix**: Add type conversion for `StringType` in `populateTypeConversion()`:
   ```cpp
   typeConverter.addConversion([&](StringType type) -> std::optional<Type> {
     // Option A: Map to LLVM pointer (like ChandleType)
     return LLVM::LLVMPointerType::get(type.getContext());
     // Option B: Reject with proper error
     return std::nullopt;  // Will trigger "failed to convert type" error
   });
   ```

2. **Better Error Handling**: Ensure that when `convertType()` fails, the error is returned before any invalid data is added to the `ports` vector.

3. **Feature Request Alternative**: If `string` ports are intentionally unsupported, add explicit documentation and a clearer error message rather than an assertion failure.

## Keywords for Issue Search
`StringType` `string` `MooreToCore` `convertType` `port` `assertion` `dyn_cast` `InOutType`

## Related Files to Investigate
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Main conversion pass, add StringType conversion
- `include/circt/Dialect/Moore/MooreTypes.td` - StringType definition
- `include/circt/Dialect/HW/HWTypes.td` - Target types for conversion
