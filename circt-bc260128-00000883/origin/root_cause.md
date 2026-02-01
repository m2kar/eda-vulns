# Root Cause Analysis: CIRCT Crash on Packed Union Type Port

## Summary

`circt-verilog` crashes with an assertion failure when processing a SystemVerilog module that has a **packed union** type as a module port. The crash occurs during the MooreToCore conversion pass when `typeConverter.convertType()` returns a null/invalid type for the packed union port, and `sanitizeInOut()` attempts to `dyn_cast<hw::InOutType>` on that null type.

## Error Context

**Assertion Failure:**
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

**Crash Location:**
- File: `include/circt/Dialect/HW/PortImplementation.h:177`
- Function: `hw::ModulePortInfo::sanitizeInOut()`

## Test Case Analysis

The test case (`source.sv`) defines:

1. A packed struct type `data_struct_t` with an 8-bit logic member
2. A packed union type `data_union_t` containing the struct and a raw logic
3. A module `data_processor` with an input port of type `data_union_t`

```systemverilog
typedef struct packed {
  logic [7:0] value;
} data_struct_t;

typedef union packed {
  data_struct_t s;
  logic [7:0] raw;
} data_union_t;

module data_processor(
  input logic clk,
  input data_union_t in_data,  // <-- Problematic port type
  output logic [7:0] result
);
```

## Stack Trace Analysis

```
#17 circt::hw::ModulePortInfo::sanitizeInOut() PortImplementation.h:177
#21 getModulePortInfo(mlir::TypeConverter const&, circt::moore::SVModuleOp) MooreToCore.cpp:259
#22 SVModuleOpConversion::matchAndRewrite() MooreToCore.cpp:276
#42 MooreToCorePass::runOnOperation() MooreToCore.cpp:2571
```

The crash path shows:
1. `MooreToCorePass` runs conversion patterns
2. `SVModuleOpConversion::matchAndRewrite()` is called for the module
3. `getModulePortInfo()` creates port information with type conversion
4. `ModulePortInfo` constructor calls `sanitizeInOut()`
5. `sanitizeInOut()` calls `dyn_cast<hw::InOutType>(p.type)` on null type → **CRASH**

## Root Cause

The root cause is a **missing null type check** in the `getModulePortInfo()` function. When `typeConverter.convertType(port.type)` fails for a packed union type (returning a null `Type`), the code does not validate the result before creating `hw::PortInfo`.

**Problematic code in MooreToCore.cpp (lines 243-258):**

```cpp
static hw::ModulePortInfo getModulePortInfo(const TypeConverter &typeConverter,
                                            SVModuleOp op) {
  // ...
  for (auto port : moduleTy.getPorts()) {
    Type portTy = typeConverter.convertType(port.type);  // May return null!
    if (port.dir == hw::ModulePort::Direction::Output) {
      ports.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, resultNum++, {}}));
    } else {
      ports.push_back(
          hw::PortInfo({{port.name, portTy, port.dir}, inputNum++, {}}));
    }
  }
  return hw::ModulePortInfo(ports);  // Calls sanitizeInOut() with null type
}
```

**The issue:** When `portTy` is null (because the type converter doesn't support packed union types), it is still used to create a `hw::PortInfo`. Later, when `ModulePortInfo`'s constructor calls `sanitizeInOut()`:

```cpp
void sanitizeInOut() {
  for (auto &p : ports)
    if (auto inout = dyn_cast<hw::InOutType>(p.type)) {  // <-- Crash: p.type is null
      p.type = inout.getElementType();
      p.dir = ModulePort::Direction::InOut;
    }
}
```

The `dyn_cast` operation fails with an assertion because it cannot operate on a null/empty `Type`.

## Why Packed Union Conversion Fails

SystemVerilog's `packed union` types need special handling during type conversion. The Moore dialect's type converter likely:

1. Doesn't have a registered conversion for packed union types, OR
2. The conversion returns failure/null for unsupported aggregate types

Packed unions are challenging because they require all members to have the same bit width, which may not map cleanly to HW dialect's type system.

## Fix Recommendations

### Option 1: Add null check in `getModulePortInfo()`

```cpp
for (auto port : moduleTy.getPorts()) {
  Type portTy = typeConverter.convertType(port.type);
  if (!portTy) {
    // Emit diagnostic or return failure
    op.emitOpError() << "failed to convert port type: " << port.type;
    return {};  // or return failure
  }
  // ... rest of the code
}
```

### Option 2: Add packed union type conversion

Implement proper conversion for Moore dialect's packed union type to HW dialect's compatible type (likely `hw::IntType` with total bit width).

### Option 3: Defensive check in `sanitizeInOut()`

```cpp
void sanitizeInOut() {
  for (auto &p : ports) {
    if (!p.type)
      continue;  // Skip null types
    if (auto inout = dyn_cast<hw::InOutType>(p.type)) {
      p.type = inout.getElementType();
      p.dir = ModulePort::Direction::InOut;
    }
  }
}
```

## Classification

| Attribute | Value |
|-----------|-------|
| **Dialect** | Moore → HW (MooreToCore pass) |
| **Crash Type** | Assertion Failure |
| **Trigger** | Packed union type as module port |
| **Severity** | High - Tool crashes on valid SystemVerilog |
| **Component** | lib/Conversion/MooreToCore/MooreToCore.cpp |

## Keywords

`packed union`, `typedef union packed`, `MooreToCore`, `getModulePortInfo`, `sanitizeInOut`, `dyn_cast`, `InOutType`, `type conversion`, `null type`, `assertion failure`
