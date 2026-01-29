# Root Cause Analysis Report

## Executive Summary

CIRCT crashes with an assertion failure when converting a Moore SVModule that contains a `string` type output port to the core dialects. The crash occurs in `sanitizeInOut()` which attempts to cast port types to `hw::InOutType`, but the port type has already been converted to `sim::DynamicStringType` (from Moore's `StringType`). The `dyn_cast` operation fails because the type is not an `InOutType` but `sanitizeInOut()` does not handle this case correctly.

## Crash Context

- **Tool/Command**: `circt-verilog --ir-hw source.sv`
- **Dialect**: Moore
- **Failing Pass**: MooreToCore (SVModuleOpConversion)
- **Crash Type**: Assertion failure

## Error Analysis

### Assertion Message
```
decltype(auto) llvm::dyn_cast(From &) [To = circt::hw::InOutType, From = mlir::Type]: Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Key Stack Frames
```
#13 (anonymous namespace)::getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp) MooreToCore.cpp:259
#12 llvm::SmallVector<circt::hw::PortInfo, 1u>::~SmallVector() llvm/ADT/SmallVector.h:1207
#11 (anonymous namespace)::getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp) MooreToCore.cpp:259
```

The crash occurs during cleanup in the destructor of `SmallVector<PortInfo>`, which suggests that the `sanitizeInOut()` function corrupted or invalidated the port types before the vector was fully destroyed.

## Test Case Analysis

### Code Summary
The test case is a simple SystemVerilog counter module with:
- Sequential logic (`always_ff`) for counter increment
- Combinational logic (`always_comb`) for status string assignment
- **A `string` type output port named `status_str`**

### Key Constructs
- `always_ff`: Moore dialect sequential block
- `always_comb`: Moore dialect combinational block
- **`string` type port**: This is the problematic construct

### Potentially Problematic Patterns
1. **String type output port**: The `status_str` port is declared as `output string`, which is not a standard hardware type
2. The Moore dialect defines `StringType` as a specific type with mnemonic `"string"`
3. In `populateTypeConversion()`, `StringType` is converted to `sim::DynamicStringType`

## CIRCT Source Analysis

### Crash Location
**File**: `include/circt/Dialect/HW/PortImplementation.h`
**Function**: `ModulePortInfo::sanitizeInOut()`
**Line**: 177

### Code Context
```cpp
// In include/circt/Dialect/HW/PortImplementation.h, lines 175-181
void sanitizeInOut() {
  for (auto &p : ports)
    if (auto inout = dyn_cast<hw::InOutType>(p.type)) {
      p.type = inout.getElementType();
      p.dir = ModulePort::Direction::InOut;
    }
}
```

### Processing Path

1. **Moore Parsing**: `source.sv` is parsed, creating a `moore.SVModuleOp` with a port of type `moore::StringType`
2. **Type Conversion** (in `MooreToCore.cpp:populateTypeConversion()`):
   - Line 2277-2279: `StringType` → `sim::DynamicStringType`
   ```cpp
   typeConverter.addConversion([&](StringType type) {
     return sim::DynamicStringType::get(type.getContext());
   });
   ```
3. **Port Info Extraction** (in `MooreToCore.cpp:getModulePortInfo()`):
   - Line 243: Port type is converted using `typeConverter.convertType(port.type)`
   - For `string` port, `portTy` becomes `sim::DynamicStringType`
   - Line 254: `PortInfo` is created with `p.type = sim::DynamicStringType`
4. **Sanitization** (in `ModulePortInfo` constructor):
   - Line 62: `sanitizeInOut()` is called on the vector of ports
   - Line 177: Attempts `dyn_cast<hw::InOutType>(p.type)`
   - **FAILS**: `p.type` is `sim::DynamicStringType`, not `hw::InOutType`

### Root Issue
The `sanitizeInOut()` function assumes all port types are either:
1. Standard hardware types (which don't match the `hw::InOutType` cast), or
2. `hw::InOutType` (which get unwrapped)

But it doesn't handle:
- Other dialect types like `sim::DynamicStringType`
- Port types that are not HW-dialect types

The assertion "dyn_cast on a non-existent value" suggests that the MLIR Type object may be in an invalid state or the `dyn_cast` operation is detecting an internal consistency issue.

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: `sanitizeInOut()` does not guard against non-HW dialect types, causing assertion failure when ports have types from other dialects (like `sim::DynamicStringType`)

**Evidence**:
- Test case has `output string status_str` which is `moore::StringType`
- `StringType` is converted to `sim::DynamicStringType` (line 2278 of MooreToCore.cpp)
- `sanitizeInOut()` performs `dyn_cast<hw::InOutType>(p.type)` without checking if `p.type` is a valid HW type
- Crash occurs at line 177 of PortImplementation.h when trying to cast non-HW type

**Mechanism**:
The function iterates through all ports and blindly attempts to cast each port's type to `hw::InOutType`. When a port type is `sim::DynamicStringType`, the `dyn_cast` detects that this is not a valid cast target and triggers an assertion failure to indicate programmer error. This is LLVM/MLIR's way of saying "you're trying to cast to an incompatible type."

### Hypothesis 2 (Medium Confidence)
**Cause**: There may be a missing type conversion or legality check that should prevent `StringType` ports from being used in contexts that assume only hardware types

**Evidence**:
- `StringType` is a SystemVerilog feature, but not all CIRCT passes may support it
- The crash happens during conversion to core dialects, suggesting string support may be incomplete

## Suggested Fix Directions

1. **Add type checking in `sanitizeInOut()`**:
   - Check if `p.type` is a valid HW type before attempting `dyn_cast`
   - Use `isa<hw::InOutType>()` first, which returns false instead of asserting
   - Skip the conversion for non-HW types

2. **Handle `sim::DynamicStringType` explicitly**:
   - Add a case in `sanitizeInOut()` to handle string ports
   - Either leave them as-is or mark them as unsupported with a proper error message

3. **Improve type conversion validation**:
   - Ensure that types from Moore dialect are properly validated before creating PortInfo
   - Add legality checks for port types during Moore to Core conversion

## Keywords for Issue Search
`string` `StringType` `DynamicStringType` `sanitizeInOut` `InOutType` `MooreToCore` `SVModuleOp` `port` `dyn_cast`

## Related Files to Investigate
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Type conversion logic
- `include/circt/Dialect/HW/PortImplementation.h` - Port info sanitization
- `include/circt/Dialect/Moore/MooreTypes.td` - String type definition
- `include/circt/Dialect/Sim/SimTypes.td` - DynamicString type definition
