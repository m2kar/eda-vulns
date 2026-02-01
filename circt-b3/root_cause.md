# Root Cause Analysis Report

## Executive Summary
CIRCT compiler crashes with an assertion failure when processing SystemVerilog modules that use packed union types as module ports. The root cause is that the MooreToCore conversion pass lacks a type conversion rule for packed union types, causing an invalid/null type to be processed during module port information extraction, which then fails an assertion in downstream type handling code.

## Crash Context
- **Tool/Command**: circt-verilog --ir-hw
- **Dialect**: Moore (SystemVerilog)
- **Failing Pass**: MooreToCore conversion
- **Crash Type**: Assertion failure
- **Assertion Message**: `dyn_cast<circt::hw::InOutType>(...) failed - "dyn_cast on a non-existent value"`

## Error Analysis

### Assertion Message
```
llvm::dyn_cast<circt::hw::InOutType>(mlir::Type)
  Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Key Stack Frames
```
#13  getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp)
     MooreToCore.cpp:259

#14  SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, ...,
     mlir::ConversionPatternRewriter&) const
     MooreToCore.cpp:276

#16  MooreToCorePass::runOnOperation()
     MooreToCore.cpp:2571
```

The crash occurs in `getModulePortInfo` when processing module ports, specifically at line 259 where `hw::PortInfo` is being constructed with port type information.

## Test Case Analysis

### Code Summary
The test case defines a packed union type and uses it as module ports:
- **Union definition**: `typedef union packed { logic [31:0] a; logic [31:0] b; } my_union;`
- **Module ports**: Both input and output ports of type `my_union`
- **Instantiation**: `Top` module instantiates `Sub` with union-typed ports

### Key Constructs
- **Packed union type**: SystemVerilog packed union with two 32-bit members
- **Module port declarations**: Using user-defined union type as module interface
- **Simple assignment**: Module passes union type through input to output

### Problematic Patterns
The critical pattern causing the crash is:
1. **User-defined packed union type** being used as **module port type**
2. This type must be converted from Moore dialect to HW dialect during `MooreToCore` conversion
3. No conversion rule exists for `UnionType`, leading to null/invalid type

## CIRCT Source Analysis

### Crash Location
**File**: `lib/Conversion/MooreToCore/MooreToCore.cpp`
**Function**: `getModulePortInfo(const TypeConverter &typeConverter, SVModuleOp op)`
**Line**: 259 (in hw::PortInfo construction)

### Code Context
```cpp
// Lines 243-260 of MooreToCore.cpp
for (auto port : moduleTy.getPorts()) {
  Type portTy = typeConverter.convertType(port.type);  // Line 244
  if (!portTy) {
    return op.emitError("failed to convert type of port '")
           << port.name << "' in module '" << op.getName() << "'";
  }
  if (port.dir == hw::ModulePort::Direction::Output) {
    ports.push_back(
        hw::PortInfo({{port.name, portTy, port.dir}, resultNum++, {}}));
  } else {
    // Line 259 - Port info constructed here
    ports.push_back(
        hw::PortInfo({{port.name, portTy, port.dir}, inputNum++, {}}));
  }
}
```

**Observation**: Line 244 checks if `portTy` is null after conversion, but the assertion failure suggests that a non-null but invalid type is being passed through, or the check is incomplete for union types.

### Processing Path
1. **SystemVerilog Parser**: Reads source.sv and creates Moore dialect with `UnionType` ports
2. **MooreToCore Pass**: Calls `SVModuleOpConversion::matchAndRewrite` for module conversion
3. **Type Conversion**: Calls `getModulePortInfo` which iterates over ports and calls `typeConverter.convertType(port.type)`
4. **TypeConverter Processing**: `convertType` searches for matching conversion rule
5. **Missing Rule**: No conversion exists for `UnionType` in `populateTypeConversion` (lines 2268-2409)
6. **Invalid Type**: The unconverted type causes issues when processed by `hw::PortInfo` or downstream operations
7. **Assertion Failure**: When `dyn_cast<InOutType>` is called on the invalid type during port processing, assertion fails

### Missing Type Conversion Rule
In `populateTypeConversion` function (MooreToCore.cpp:2268-2409), the following types have conversion rules:
- ✅ IntType → IntegerType
- ✅ RealType → Float32Type/Float64Type
- ✅ TimeType → llhd::TimeType
- ✅ FormatStringType → sim::FormatStringType
- ✅ ArrayType → hw::ArrayType
- ✅ UnpackedArrayType → hw::ArrayType
- ✅ StructType → hw::StructType
- ✅ UnpackedStructType → hw::StructType
- ✅ CHandleType → LLVM::LLVMPointerType
- ✅ ClassHandleType → LLVM::LLVMPointerType
- ✅ RefType → llhd::RefType
- ❌ **UnionType → [MISSING]**
- ❌ **UnpackedUnionType → [MISSING]**

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: MooreToCore conversion pass lacks type conversion rule for packed union types (`UnionType`), causing assertion failure when union-typed ports are processed.

**Evidence**:
- Test case explicitly uses `typedef union packed` as module port type
- Stack trace shows crash in `getModulePortInfo` during port processing
- Assertion message indicates `dyn_cast<InOutType>` failed on non-existent value
- No conversion rule for `UnionType` exists in `populateTypeConversion` function
- Similar types like `StructType` have conversion rules, but `UnionType` does not

**Mechanism**:
The MooreToCore type converter iterates over module port types and calls `convertType()` for each. When it encounters a `UnionType`:
1. The typeConverter searches for a matching conversion rule
2. No rule is found for UnionType (no `addConversion` for it)
3. The converter may return a null type, or create an invalid/uninitialized type
4. This invalid type is then used to construct `hw::PortInfo`
5. When port info is processed (e.g., in `fnToMod` function that calls `dyn_cast<InOutType>`), the assertion fails because the type is null or invalid

**Why this works for structs but not unions**:
- `StructType` has explicit conversion to `hw::StructType` (lines 2306-2317)
- `UnionType` is structurally similar but lacks a conversion rule
- Both implement `DestructurableTypeInterface`, suggesting similar processing should work

### Hypothesis 2 (Medium Confidence)
**Cause**: The type conversion may be returning a partially converted or malformed type instead of null, bypassing the null check at line 245-248.

**Evidence**:
- Line 245-248 checks `if (!portTy)` and emits error
- However, the crash still occurs, suggesting either:
  - The check is not being reached
  - The type is non-null but invalid (e.g., incorrect type ID)
  - The assertion occurs in a different code path not covered by the check

**Mechanism**:
If the typeConverter returns a non-null type with incorrect MLIR type ID or metadata:
1. The null check at line 245 passes (type is not null)
2. Port info is constructed with the invalid type
3. Later operations try to use the type and fail with dyn_cast assertion

### Hypothesis 3 (Low Confidence)
**Cause**: Packed unions may require special handling in HW dialect that's not implemented, causing type incompatibility even if conversion succeeds.

**Evidence**:
- HW dialect has `hw::StructType` but no explicit `hw::UnionType`
- Packed unions in SystemVerilog have special semantics (overlapping storage)
- May need to be represented differently in HW dialect

**Mechanism**:
If union types should be lowered to struct types or other representation:
1. Missing conversion rule leaves them unconverted
2. Or incorrect conversion creates type mismatch
3. HW dialect operations expecting struct types fail when encountering union types

## Suggested Fix Directions

### Direction 1: Add Type Conversion Rule for Packed Unions (Recommended)
Add a conversion rule for `UnionType` in `populateTypeConversion` function:

```cpp
typeConverter.addConversion([&](UnionType type) -> std::optional<Type> {
  // Option 1: Convert to struct type with same members
  SmallVector<hw::StructType::FieldInfo> fields;
  for (auto member : type.getMembers()) {
    hw::StructType::FieldInfo info;
    info.type = typeConverter.convertType(member.type);
    if (!info.type)
      return {};
    info.name = member.name;
    fields.push_back(info);
  }
  return hw::StructType::get(type.getContext(), fields);

  // Option 2: Represent as packed array of appropriate size
  // (Calculate union size and create hw::PackedArray)
  // uint32_t unionSize = ...;
  // return hw::PackedArray::get(...);

  // Option 3: Reject with proper error message
  // return op.emitError("packed union types are not supported in module ports");
});
```

**Rationale**: This provides a clear conversion path similar to `StructType`. Option 1 (convert to struct) is simplest and preserves type information.

### Direction 2: Add Unpacked Union Conversion
Similarly add conversion for `UnpackedUnionType`:

```cpp
typeConverter.addConversion([&](UnpackedUnionType type) -> std::optional<Type> {
  // Similar to packed union conversion
  // ... (implementation similar to UnionType conversion)
});
```

### Direction 3: Add Early Type Validation
Enhance the null check at line 245 to catch more error cases:

```cpp
Type portTy = typeConverter.convertType(port.type);
if (!portTy || portTy.isa<NoneType>()) {
  return op.emitError("failed to convert type of port '")
         << port.name << "' in module '" << op.getName()
         << "': unsupported type '" << port.type << "'";
}
```

**Rationale**: Provides better error messages even if type conversion returns invalid types.

### Direction 4: Add Runtime Type Validation in Port Processing
Add validation before port info construction to catch invalid types earlier:

```cpp
// In hw::ModulePortInfo constructor or related functions
assert(type && "Port type cannot be null");
```

Or more gracefully:
```cpp
if (!type)
  return emitError("invalid port type");
```

## Keywords for Issue Search
`packed union` `union type` `module port` `MooreToCore` `type conversion` `dyn_cast` `InOutType` `SVModuleOp` `UnionType`

## Related Files to Investigate
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Main conversion logic, missing UnionType conversion
- `lib/Dialect/HW/HWTypes.cpp` - HW dialect type definitions, InOutType handling (lines 1066, 1073)
- `lib/Dialect/HW/HWInstanceImplementation.cpp` - Port info processing, dyn_cast<InOutType> usage (line 375)
- `include/circt/Dialect/Moore/MooreTypes.td` - UnionType TableGen definition
- `include/circt/Dialect/Moore/MooreOps.td` - SVModuleOp definition

## Test Case Verification
The test case is valid SystemVerilog code:
- Packed unions are a valid IEEE 1800-2005 feature
- Using unions as module ports is syntactically valid
- The crash is a compiler bug, not invalid test case

**IEEE 1800-2005 Reference**: Section 7.3 (Packed structures and unions) defines packed unions as valid constructs.

## Impact Assessment
- **Severity**: Medium (compiler crash prevents code compilation)
- **Affected Components**: circt-verilog with --ir-hw flag
- **Affected Code Patterns**: Any module using packed/unpacked union types as ports
- **Workaround**: Avoid using union types as module ports, or use struct types instead
- **Similar Issues**: May affect other operations that process union types in Moore dialect

## Crash Category
**Category**: Null/Invalid Value Access
- Type: Assertion failure due to dyn_cast on invalid type object
- Root cause: Missing type conversion rule causing invalid type propagation
