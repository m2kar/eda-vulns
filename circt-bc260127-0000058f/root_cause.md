# Root Cause Analysis Report

## Crash ID: bc260127-0000058f

## Summary

The crash occurs in circt-verilog when processing a SystemVerilog module with **packed union** types as port types. The MooreToCore conversion pass lacks type conversion support for `PackedUnionType`, resulting in a null type being passed to `hw::ModulePortInfo`, which then triggers an assertion failure when attempting `dyn_cast` on a non-existent value.

## Crash Details

- **Crash Type:** Assertion failure
- **Assertion Message:** `detail::isPresent(Val) && "dyn_cast on a non-existent value"`
- **Tool:** circt-verilog
- **Pass:** MooreToCore conversion pass
- **Dialect:** Moore (converting to HW/Core dialects)

## Stack Trace Analysis

The crash originates from the following call chain:

```
#17 circt::hw::ModulePortInfo::sanitizeInOut()
    @ PortImplementation.h:177
    
#21 getModulePortInfo()
    @ MooreToCore.cpp:259
    
#22 SVModuleOpConversion::matchAndRewrite()
    @ MooreToCore.cpp:276
    
#42 MooreToCorePass::runOnOperation()
    @ MooreToCore.cpp:2571
```

### Key Function: `sanitizeInOut()` (PortImplementation.h:175-181)

```cpp
void sanitizeInOut() {
  for (auto &p : ports)
    if (auto inout = dyn_cast<hw::InOutType>(p.type)) {  // LINE 177 - CRASH
      p.type = inout.getElementType();
      p.dir = ModulePort::Direction::InOut;
    }
}
```

The crash occurs because `p.type` is **null** (invalid), and `dyn_cast` on a null type triggers the assertion.

### Key Function: `getModulePortInfo()` (MooreToCore.cpp:233-259)

```cpp
static hw::ModulePortInfo getModulePortInfo(const TypeConverter &typeConverter,
                                            SVModuleOp op) {
  // ...
  for (auto port : moduleTy.getPorts()) {
    Type portTy = typeConverter.convertType(port.type);  // Returns NULL for PackedUnionType!
    // portTy is used without null check...
    ports.push_back(hw::PortInfo({{port.name, portTy, port.dir}, ...}));
  }
  return hw::ModulePortInfo(ports);  // sanitizeInOut() called in constructor
}
```

## Source Code Analysis

### Input File: source.sv

```systemverilog
typedef union packed {
  logic [7:0] byte_data;
  logic [3:0][1:0] nibble_pairs;
} my_union;

module union_register_module(
  input logic clk,
  input logic rst_n,
  input my_union data_in,    // <-- PackedUnionType port
  output my_union data_out   // <-- PackedUnionType port
);
  // ...
endmodule
```

The problematic construct is using a **packed union type** (`my_union`) as module port types. This is valid SystemVerilog but not supported in the MooreToCore conversion.

## Root Cause

### Primary Issue: Missing Type Conversion for `PackedUnionType`

The MooreToCore type converter (MooreToCore.cpp:2256-2381) registers conversions for many types but **does NOT include `PackedUnionType`**:

- `IntType` - Supported
- `ArrayType` - Supported  
- `StructType` - Supported
- `UnpackedStructType` - Supported
- `PackedUnionType` - **NOT SUPPORTED** (missing conversion)

When `typeConverter.convertType()` is called on a `PackedUnionType`, it returns a **null Type** because no conversion is registered.

### Secondary Issue: No Null Check

The `getModulePortInfo()` function does not check if `typeConverter.convertType()` returns null before using the result. This allows the invalid type to propagate to `ModulePortInfo`, causing the crash.

## Affected Components

1. **MooreToCore.cpp** - `getModulePortInfo()` function (lines 233-259)
2. **MooreToCore.cpp** - Type converter setup (lines 2256-2381)
3. **PortImplementation.h** - `sanitizeInOut()` function (line 177)

## Trigger Pattern

Any SystemVerilog module with a **packed union type** as an input or output port will trigger this crash when using `circt-verilog --ir-hw`.

### Minimal Triggering Pattern

```systemverilog
typedef union packed { logic [7:0] a; } u_t;
module m(input u_t x);
endmodule
```

## Recommended Fix

1. **Add PackedUnionType conversion** to the MooreToCore type converter:
   - Convert to an integer type of appropriate bit width (union members share storage)
   - Or convert to `hw::StructType` with bitcast semantics

2. **Add null check** in `getModulePortInfo()`:
   ```cpp
   Type portTy = typeConverter.convertType(port.type);
   if (!portTy) {
     // Emit error or use fallback type
     return failure();
   }
   ```

## Classification

- **Bug Type:** Missing feature / Incomplete type conversion
- **Severity:** Crash (assertion failure)
- **User Impact:** Cannot compile valid SystemVerilog with packed union port types
