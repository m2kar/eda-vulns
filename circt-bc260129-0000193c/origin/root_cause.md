# Root Cause Analysis Report

## Summary

**Crash Type**: Assertion Failure  
**Dialect**: LLHD (Low-Level Hardware Description)  
**Pass**: Mem2Reg (Memory-to-Register Promotion)  
**Crash Location**: `lib/Dialect/LLHD/Transforms/Mem2Reg.cpp:1742` in `Promoter::insertBlockArgs`

## Error Context

```
<unknown>:0: error: integer bitwidth is limited to 16777215 bits
Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

The crash occurs when attempting to create an MLIR `IntegerType` with a bitwidth exceeding the maximum allowed value of 16777215 bits (0xFFFFFF).

## Test Case Analysis

The test case (`source.sv`) contains SystemVerilog code using `real` (64-bit floating-point) types:

```systemverilog
module test_module(
  input logic clk,
  input real in_real,
  output real out_real,
  output logic cmp_result
);

  always_ff @(posedge clk) begin
    out_real <= in_real * 2.0;
    cmp_result <= (in_real > 5.0);
  end

endmodule
```

Key observations:
- Uses `real` type for input/output ports
- Uses `always_ff` (sequential logic block with clock)
- Involves floating-point arithmetic (`* 2.0`) and comparison (`> 5.0`)

## Technical Root Cause

### Call Stack Analysis

```
#12 mlir::IntegerType::get(mlir::MLIRContext*, unsigned int, ...)
#13 Promoter::insertBlockArgs(BlockEntry*) [Mem2Reg.cpp:1742]
#14 Promoter::insertBlockArgs()            [Mem2Reg.cpp:1654]
#15 Promoter::promote()                    [Mem2Reg.cpp:764]
#16 Mem2RegPass::runOnOperation()          [Mem2Reg.cpp:1844]
```

### Root Cause Hypothesis

The crash occurs due to **improper handling of floating-point types** (`Float64Type` for `real`) in the LLHD Mem2Reg pass:

1. **Type Conversion Issue**: When the SystemVerilog `real` type is imported via the Moore dialect and converted to core dialects through MooreToCore, it becomes `Float64Type` (MLIR's native 64-bit floating-point type).

2. **Bitwidth Calculation Failure**: The `hw::getBitWidth()` function in `lib/Dialect/HW/HWTypes.cpp` only handles `IntegerType` for built-in types:
   ```cpp
   return llvm::TypeSwitch<::mlir::Type, int64_t>(type)
       .Case<IntegerType>(
           [](IntegerType t) { return t.getIntOrFloatBitWidth(); })
       .Default([](Type type) -> int64_t {
           // Returns -1 for unknown types including Float64Type
           if (auto iface = dyn_cast<BitWidthTypeInterface>(type)) {
               std::optional<int64_t> width = iface.getBitWidth();
               return width.has_value() ? *width : -1;
           }
           return -1;
       });
   ```

3. **Integer Type Creation with Invalid Width**: The Mem2Reg pass's `insertBlockArgs` function attempts to create block arguments for promoted memory slots. When processing a slot with a floating-point type:
   - It likely uses `hw::getBitWidth()` or similar to determine the type width
   - For `Float64Type`, this returns -1 (unknown)
   - The -1 value (when interpreted as unsigned) becomes a very large number, or the code incorrectly tries to create an `IntegerType` for a non-integer slot type
   - This results in attempting to create an `IntegerType` with an invalid bitwidth exceeding the 16777215 limit

4. **Missing Type Support**: The Mem2Reg pass was designed primarily for integer/bit-vector types typical in hardware designs and does not properly handle or reject floating-point types.

### The Bug

The LLHD Mem2Reg pass fails to:
1. Properly validate that slot types are promotable (integer/bit-vector types only)
2. Handle or gracefully reject floating-point types before attempting promotion
3. Check the result of bitwidth calculations for validity before creating IntegerTypes

## Impact

- **Severity**: High (assertion failure causing compiler crash)
- **Reproducibility**: Deterministic - occurs whenever `real` type signals are used in sequential logic blocks (`always_ff`)
- **Scope**: Affects any SystemVerilog design using floating-point types with the `circt-verilog --ir-hw` pipeline

## Suggested Fix

1. **Short-term**: Add type validation in the Mem2Reg pass to skip or reject non-promotable types (floating-point, strings, etc.)

2. **Long-term**: 
   - Extend `hw::getBitWidth()` to properly handle MLIR floating-point types
   - Add proper support for floating-point register promotion in Mem2Reg, or explicitly exclude them from promotion with appropriate diagnostics

## Related Components

- `lib/Dialect/LLHD/Transforms/Mem2Reg.cpp` - Crash location
- `lib/Dialect/HW/HWTypes.cpp` - `getBitWidth()` function
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - Type conversion for `RealType`
- `lib/Conversion/ImportVerilog/Types.cpp` - SystemVerilog type import

## References

- Related Issue: [#8269 - [MooreToCore] Support `real` constants](https://github.com/llvm/circt/issues/8269)
- Related Issue: [#8930 - [MooreToCore] Crash with sqrt/floor](https://github.com/llvm/circt/issues/8930)
