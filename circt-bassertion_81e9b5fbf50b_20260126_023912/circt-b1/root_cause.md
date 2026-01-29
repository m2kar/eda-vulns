# Root Cause Analysis Report

## Executive Summary

CIRCT crashes when compiling SystemVerilog code that uses packed unions as module ports. The bug is caused by missing type conversion logic for UnionType in the MooreToCore pass. When a module has a packed union type port, the type converter fails to translate it from Moore dialect to HW dialect, resulting in a null type value that triggers an assertion failure when subsequent code attempts to perform `dyn_cast<InOutType>`.

## Crash Context

- **Tool/Command**: `circt-verilog --ir-hw`
- **Dialect**: Moore (SystemVerilog)
- **Failing Pass**: MooreToCore (Moore to HW dialect conversion)
- **Crash Type**: Assertion failure (attempting dyn_cast on null value)
- **CIRCT Version**: 1.139.0 (original crash), 22.0.0git (reproduction)

## Error Analysis

### Assertion/Error Message

```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
llvm::dyn_cast<circt::hw::InOutType, From = mlir::Type>
```

### Key Stack Frames

```
#13 (anonymous namespace)::getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp)
    at MooreToCore.cpp:259

#14 (anonymous namespace)::SVModuleOpConversion::matchAndRewrite(circt::moore::SVModuleOp, ...)
    at MooreToCore.cpp:276
```

**Crash Location**: `MooreToCore.cpp:259` in `getModulePortInfo()` function

## Test Case Analysis

### Code Summary

The test case demonstrates a SystemVerilog design using packed unions as module ports:

```systemverilog
typedef union packed {
  logic [31:0] a;
  logic [31:0] b;
} my_union;

module mod1(output my_union out);
  assign out.a = 32'h1234_5678;
endmodule

module mod2(input my_union in);
  logic [32:0] val;
  assign val = in.b;
endmodule

module top();
  my_union conn;
  mod1 m1(.out(conn));
  mod2 m2(.in(conn));
endmodule
```

**Purpose**: Tests instantiation and connection of modules with packed union type ports.

### Key Constructs

- **packed union**: A union type where members share the same memory space (packed)
- **Module port of union type**: Using my_union as input/output port type
- **Module instantiation**: Connecting union signals between modules

### Potentially Problematic Patterns

- **Union type in port declarations**: The use of packed unions as module ports is not handled by MooreToCore type conversion
- **Union field access**: `out.a` and `in.b` - accessing fields of union typed ports

## CIRCT Source Analysis

### Crash Location

**File**: `lib/Conversion/MooreToCore/MooreToCore.cpp`
**Function**: `static hw::ModulePortInfo getModulePortInfo(const TypeConverter &typeConverter, SVModuleOp op)`
**Line**: 259 (approximately in the port processing loop)

### Code Context

```cpp
// From MooreToCore.cpp:234-254
static hw::ModulePortInfo getModulePortInfo(const TypeConverter &typeConverter,
                                            SVModuleOp op) {
  size_t inputNum = 0;
  size_t resultNum = 0;
  auto moduleTy = op.getModuleType();
  SmallVector<hw::PortInfo> ports;
  ports.reserve(moduleTy.getNumPorts());

  for (auto port : moduleTy.getPorts()) {
    Type portTy = typeConverter.convertType(port.type);  // ← FAILS HERE for UnionType
    if (port.dir == hw::ModulePort::Direction::Output) {
      ports.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, resultNum++, {}}));
    } else {
      // FIXME comment about not supporting inout/ref ports yet
      ports.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, inputNum++, {}}));
    }
  }

  return hw::ModulePortInfo(ports);
}
```

### Processing Path

1. **Parser**: circt-verilog parses SystemVerilog and creates Moore dialect IR
   - Packed union `my_union` is represented as `UnionType` in Moore dialect
   - Modules have ports of type `UnionType`

2. **Type Conversion Setup**: In `MooreToCorePass::runOnOperation()` (line 2605):
   ```cpp
   TypeConverter typeConverter;
   populateTypeConversion(typeConverter);  // ← MISSING UnionType conversion
   ```

3. **Type Conversion Registration** (`populateTypeConversion` function, line 2255):
   - ✅ IntType → IntegerType
   - ✅ RealType → Float32/Float64Type
   - ✅ StructType → hw::StructType
   - ✅ UnpackedStructType → hw::StructType
   - ✅ ArrayType, UnpackedArrayType → hw::ArrayType
   - ❌ **UnionType → ???** (NOT REGISTERED)
   - ❌ **UnpackedUnionType → ???** (NOT REGISTERED)

4. **Module Conversion** (`SVModuleOpConversion::matchAndRewrite`, line 276):
   - Calls `getModulePortInfo()` to extract port information
   - For each port, calls `typeConverter.convertType(port.type)`

5. **Failure Point** (line 244):
   - When `port.type` is `UnionType`, there is no registered conversion
   - `convertType()` returns null or fails
   - `portTy` becomes null
   - Subsequent code attempts `dyn_cast<InOutType>(portTy)` or similar
   - Assertion fails: "dyn_cast on a non-existent value"

### Supporting Evidence

**Moore dialect type definitions** (`include/circt/Dialect/Moore/MooreTypes.td`):
```tablegen
def UnionType : StructLikeType<"Union", [...]
def UnpackedUnionType : StructLikeType<"UnpackedUnion", [...]
def AnyUnionType : MooreType<
  Or<[UnionType.predicate, UnpackedUnionType.predicate]>
```

These types are defined in Moore dialect but **no conversion exists** for them in MooreToCore.

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence): Missing UnionType Type Conversion

**Cause**: The MooreToCore pass type converter lacks a conversion rule for UnionType and UnpackedUnionType, causing conversion to fail when union-typed ports are encountered.

**Evidence**:
- Test case uses packed union as module port type
- Stack trace shows crash in `getModulePortInfo()` during port processing
- Code review reveals no `addConversion` for UnionType/UnpackedUnionType in `populateTypeConversion()` (lines 2255-2400)
- Similar types like StructType and UnpackedStructType have conversions defined
- Assertion message indicates `dyn_cast` on a null value, consistent with failed type conversion

**Mechanism**:
```
Input: module mod1(output my_union out) → Moore IR has UnionType port
  ↓
MooreToCore::getModulePortInfo() calls typeConverter.convertType(UnionType)
  ↓
Type converter has no UnionType handler → returns null/undefined
  ↓
portTy = null/undefined
  ↓
Subsequent code tries dyn_cast<InOutType>(portTy)
  ↓
Assertion: "dyn_cast on a non-existent value"
  ↓
CRASH
```

**Confidence**: High - The evidence strongly supports this hypothesis. Missing type conversion is a common cause of such assertion failures in MLIR dialect conversion passes.

### Hypothesis 2 (Low Confidence): Incorrect Port Direction Handling

**Cause**: The bug might be related to how the pass handles port directions for union types, potentially attempting to treat them as inout ports when they're actually input/output.

**Evidence**:
- The assertion involves `InOutType`, suggesting some code path attempts to cast port type to InOutType
- Code has FIXME comment about not supporting inout/ref ports (line 247)
- However, the test case explicitly uses input/output, not inout

**Mechanism**: Possibly, when type conversion fails, some fallback logic incorrectly assumes union types must be inout ports.

**Confidence**: Low - This doesn't explain why type conversion would fail in the first place. Hypothesis 1 is more fundamental.

## Suggested Fix Directions

1. **Add UnionType Type Conversion** (Primary Fix):
   - Add conversion for `UnionType` to `hw::StructType` in `populateTypeConversion()` function
   - Similar to existing StructType conversion:
   ```cpp
   typeConverter.addConversion([&](moore::UnionType type) -> std::optional<Type> {
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
   });
   ```

2. **Add UnpackedUnionType Type Conversion** (Secondary Fix):
   - Similar conversion for unpacked unions, possibly also to `hw::StructType`
   - Note: May need special handling for unpacked vs packed semantics

3. **Add Type Conversion Validation** (Robustness):
   - Before using converted type, check if it's valid
   - Add error message if type conversion fails instead of crashing:
   ```cpp
   Type portTy = typeConverter.convertType(port.type);
   if (!portTy) {
     op.emitError() << "unsupported port type: " << port.type;
     return {};
   }
   ```

4. **Improve Error Messages** (Developer Experience):
   - Replace assertion with proper diagnostic
   - Help users understand that union types in ports might not be fully supported

## Keywords for Issue Search

`packed union` `union type` `module port` `MooreToCore` `type conversion` `UnionType` `InOutType`

## Related Files to Investigate

- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Main conversion pass, add type conversion here
- `include/circt/Dialect/Moore/MooreTypes.td` - Moore dialect type definitions
- `include/circt/Dialect/HW/HWTypes.td` - HW dialect type definitions (InOutType)
- `include/circt/Dialect/HW/HWOps.td` - HW dialect operation definitions
- Test files for similar type conversions in `test/Conversion/MooreToCore/`

## Test Cases for Fix Validation

After implementing the fix, test with:

1. **Basic union port**: Current test case
2. **Nested union**: Unions containing structs or other unions
3. **Union arrays**: Array of union types
4. **Union in submodule**: Hierarchical module instantiation
5. **Unpacked union**: Using `union unpacked` instead of `union packed`
