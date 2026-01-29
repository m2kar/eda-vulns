# Root Cause Analysis Report

## Executive Summary

CIRCT `circt-verilog` crashes with an assertion failure when processing a SystemVerilog module that uses `string` type as a port. The root cause is that the Moore-to-Core type conversion does not handle `string` types, returning an invalid/null type that causes a subsequent `dyn_cast` assertion to fail.

## Crash Context

| Field | Value |
|-------|-------|
| Tool | circt-verilog |
| Dialect | Moore (SystemVerilog frontend) |
| Failing Pass | MooreToCorePass |
| Crash Type | Assertion failure |
| CIRCT Version | 1.139.0 |

## Error Analysis

### Assertion Message
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

### Crash Location
- **File**: `MooreToCore.cpp:259`
- **Function**: `getModulePortInfo()`
- **Actual failure**: `PortImplementation.h:177` in `sanitizeInOut()`

### Key Stack Frames
```
#17 circt::hw::ModulePortInfo::sanitizeInOut() PortImplementation.h:177
#21 (anonymous namespace)::getModulePortInfo() MooreToCore.cpp:259
#22 SVModuleOpConversion::matchAndRewrite() MooreToCore.cpp:276
#42 MooreToCorePass::runOnOperation() MooreToCore.cpp:2571
```

## Test Case Analysis

### Code Summary
```systemverilog
module sub_module(
  input logic clock,
  output logic [7:0] out,
  input string msg         // <-- Problematic: string as port type
);
  always @(posedge clock) begin
    out <= 8'h00;
  end
endmodule
```

### Key Constructs
- `string` type used as module input port
- `string` is a SystemVerilog dynamic class type (IEEE 1800-2017)
- Not synthesizable, primarily for simulation/verification

### Problematic Pattern
Using `string` as a module port type is legal SystemVerilog syntax but not synthesizable. The Moore dialect can represent this, but the HW dialect cannot, as HW only supports hardware-synthesizable types.

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence) ⭐

**Cause**: Moore-to-Core type conversion fails silently for `string` type, producing an invalid type that crashes later

**Evidence**:
1. Test declares `input string msg` which Moore dialect parses successfully
2. During SVModuleOp conversion, `getModulePortInfo()` calls the TypeConverter
3. TypeConverter has no rule for `moore::StringType` → returns null/invalid type
4. The port info is constructed with this invalid type
5. `ModulePortInfo::sanitizeInOut()` attempts `dyn_cast<hw::InOutType>` on the invalid type
6. Assertion fires because `detail::isPresent(Val)` returns false for null/invalid type

**Mechanism**:
```
moore.module with StringType port
    ↓ (MooreToCore conversion)
TypeConverter::convertType(StringType) → returns null (no conversion rule)
    ↓
PortInfo created with invalid type
    ↓
sanitizeInOut() → dyn_cast<InOutType>(invalid_type) → ASSERTION FAILURE
```

### Hypothesis 2 (Medium Confidence)

**Cause**: Missing type conversion rule in MooreToCore type converter

**Evidence**:
- Moore dialect supports `StringType` for SystemVerilog compatibility
- MooreToCoreTypeConverter doesn't have explicit handling for non-synthesizable types
- Should either convert to a placeholder type or emit a proper error diagnostic

## Suggested Fix Directions

1. **Proper Error Handling (Recommended)**:
   - Add type conversion rule that fails with a clear diagnostic for unsupported types
   - "String type ports are not supported for synthesis"
   
2. **Graceful Degradation**:
   - Convert `string` to an opaque type or placeholder
   - Allow simulation-only constructs to pass through with warnings

3. **Early Validation**:
   - Add a pre-pass that validates all port types are synthesizable
   - Emit user-friendly error before conversion begins

## Keywords for Issue Search
`string` `port` `MooreToCore` `type conversion` `StringType` `InOutType` `dyn_cast` `assertion` `getModulePortInfo`

## Related Files
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Conversion pass and type converter
- `include/circt/Dialect/HW/PortImplementation.h` - Port handling utilities
- `include/circt/Dialect/Moore/MooreTypes.h` - Moore type definitions
