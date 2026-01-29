# Root Cause Analysis Report

## Executive Summary

circt-verilog crashes when processing SystemVerilog code with **packed union types** used as module port types. The MooreToCore conversion pass lacks a type converter for `moore::UnionType`, causing `typeConverter.convertType()` to return a null type. This null type is then used in a `dyn_cast<hw::InOutType>()` call, triggering the assertion "dyn_cast on a non-existent value".

## Crash Context

| Field | Value |
|-------|-------|
| Tool | circt-verilog |
| Dialect | Moore |
| Failing Pass | MooreToCorePass |
| Crash Type | Assertion failure |
| CIRCT Version | 1.139.0 |

## Error Analysis

### Assertion Message
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Crash Location
- **File**: `lib/Conversion/MooreToCore/MooreToCore.cpp:259`
- **Function**: `getModulePortInfo()`
- **Called from**: `SVModuleOpConversion::matchAndRewrite()` (line 276)

### Key Stack Frames
```
#11 getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp)
    MooreToCore.cpp:259
#12 SVModuleOpConversion::matchAndRewrite()
    MooreToCore.cpp:276
#13 mlir::ConversionPattern::dispatchTo1To1<>()
#35 MooreToCorePass::runOnOperation()
    MooreToCore.cpp:2571
```

## Test Case Analysis

### Code Summary
```systemverilog
typedef union packed {
  logic [31:0] a;
  logic [31:0] b;
} my_union;

module Sub(input my_union in_val, output my_union out_val);
  assign out_val = in_val;
endmodule

module Top;
  my_union data_in, data_out;
  Sub s(.in_val(data_in), .out_val(data_out));
endmodule
```

The test case:
1. Defines a **packed union** type `my_union`
2. Uses `my_union` as **module port types** (both input and output)
3. Instantiates the module and connects union-typed signals

### Key Constructs
- `union packed` - SystemVerilog packed union type
- Module ports with user-defined type (`my_union`)
- Type used in port declarations (input/output)

### Problematic Pattern
The packed union type `my_union` is used as a port type. When MooreToCore tries to convert the module ports, it attempts to convert this type but fails because no converter is registered for `moore::UnionType`.

## CIRCT Source Analysis

### Type Conversion Registration (MooreToCore.cpp:2255-2380)

The `populateTypeConversion()` function registers converters for various Moore types:

| Moore Type | Target Type | Status |
|------------|-------------|--------|
| `IntType` | `IntegerType` | Supported |
| `ArrayType` | `hw::ArrayType` | Supported |
| `StructType` | `hw::StructType` | Supported |
| `UnpackedStructType` | `hw::StructType` | Supported |
| `RefType` | `llhd::RefType` | Supported |
| **`UnionType`** | - | **MISSING** |
| **`UnpackedUnionType`** | - | **MISSING** |

### Crash Mechanism

1. **Input**: Module `Sub` has ports of type `my_union` (a packed union)
2. **Moore IR**: Port type is represented as `moore::UnionType`
3. **getModulePortInfo()** (line 243):
   ```cpp
   Type portTy = typeConverter.convertType(port.type);
   ```
4. **Type conversion fails**: No converter for `UnionType` returns null
5. **Later code** attempts to use this null type:
   ```cpp
   dyn_cast<hw::InOutType>(portTy)  // portTy is null!
   ```
6. **Assertion fires**: "dyn_cast on a non-existent value"

### Moore Union Type Definition (MooreTypes.td)

```tablegen
def UnionType : StructLikeType<"Union", [...]>
def UnpackedUnionType : StructLikeType<"UnpackedUnion", [...]>
```

Both types exist in the Moore dialect but lack corresponding type converters in MooreToCore.

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence): Missing UnionType Converter

**Description**: The MooreToCore pass does not implement type conversion for `moore::UnionType` and `moore::UnpackedUnionType`.

**Evidence**:
1. Source grep shows no `UnionType` in `addConversion` calls
2. `StructType` and `UnpackedStructType` have converters, but unions do not
3. Stack trace shows crash in type conversion path
4. Test case uses packed union as port type

**Confidence**: **95%**

**Impact**: Any use of packed/unpacked unions in module ports will crash.

### Hypothesis 2 (Medium Confidence): Incomplete Feature Implementation

**Description**: Union type support may have been partially implemented in Moore dialect parsing but not completed in the lowering pass.

**Evidence**:
1. `UnionType` exists in MooreTypes.td (defined alongside `StructType`)
2. Struct types have full conversion support
3. Common pattern in incremental feature development

**Confidence**: **80%**

## Suggested Fix Directions

### Option 1: Add Union Type Converter (Recommended)

Add type conversion for `UnionType` similar to `StructType`:

```cpp
// In populateTypeConversion()
typeConverter.addConversion([&](UnionType type) -> std::optional<Type> {
  // Option A: Convert to struct (all members overlay same memory)
  // This loses union semantics but enables basic simulation
  SmallVector<hw::StructType::FieldInfo> fields;
  for (auto field : type.getMembers()) {
    hw::StructType::FieldInfo info;
    info.type = typeConverter.convertType(field.type);
    if (!info.type)
      return {};
    info.name = field.name;
    fields.push_back(info);
  }
  return hw::StructType::get(type.getContext(), fields);
  
  // Option B: Convert to largest member type (true union semantics)
  // return IntegerType::get(type.getContext(), type.getBitSize());
});
```

### Option 2: Add Proper Error Handling

If union support is intentionally not implemented, add a clear error message:

```cpp
typeConverter.addConversion([&](UnionType type) -> std::optional<Type> {
  return std::nullopt;  // Will trigger proper MLIR conversion failure
});
```

And handle the null type in `getModulePortInfo()`:
```cpp
Type portTy = typeConverter.convertType(port.type);
if (!portTy) {
  // Emit diagnostic and return failure
}
```

### Option 3: Emit Unsupported Feature Diagnostic

Add explicit check before conversion:
```cpp
if (isa<UnionType>(port.type)) {
  op.emitError("packed union port types not yet supported");
  return failure();
}
```

## Keywords for Issue Search
`UnionType` `packed union` `MooreToCore` `type conversion` `module port` `dyn_cast`

## Related Files
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Main conversion pass
- `include/circt/Dialect/Moore/MooreTypes.h` - Moore type definitions
- `include/circt/Dialect/Moore/MooreTypes.td` - Moore type TableGen
- `include/circt/Dialect/HW/HWTypes.h` - Target HW types

## IEEE 1800 Reference
- Section 7.3: Packed unions
- Section 7.3.1: Packed union members must be of packed types
