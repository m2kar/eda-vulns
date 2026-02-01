# Root Cause Analysis: CIRCT Crash in LLHD Mem2Reg Pass

## Summary

The crash occurs in CIRCT's LLHD dialect Mem2Reg pass when attempting to create an MLIR `IntegerType` with a bit width exceeding the MLIR limit of 16,777,215 bits. This happens when processing SystemVerilog `real` (floating-point) types through the Mem2Reg optimization pass.

## Error Context

**Error Message:**
```
<unknown>:0: error: integer bitwidth is limited to 16777215 bits
```

**Assertion Failed:**
```
Assertion `succeeded(ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

**Location:** `mlir/include/mlir/IR/StorageUniquerSupport.h:180`

## Stack Trace Analysis

The crash path is:
1. `circt-verilog` processes SystemVerilog input with `--ir-hw` flag
2. The LLHD `Mem2RegPass::runOnOperation()` is invoked
3. `Promoter::promote()` is called to perform memory-to-register promotion
4. `Promoter::insertBlockArgs()` → `insertBlockArgs(BlockEntry*)` iterates block entries
5. At line 1742-1753 in `Mem2Reg.cpp`, when creating a default value for an uninitialized slot:
   ```cpp
   auto type = getStoredType(slot);
   auto flatType = builder.getIntegerType(hw::getBitWidth(type));
   ```
6. `hw::getBitWidth()` returns -1 for unsupported types (like `real`)
7. `-1` is interpreted as an unsigned integer, becoming an extremely large value
8. `builder.getIntegerType(-1)` tries to create an IntegerType with ~2^64-1 bits
9. MLIR's IntegerType verifier fails because the width exceeds 16,777,215 bits

## Input Analysis

The test case `source.sv` contains:
```systemverilog
module test_module(
  input logic clk,
  input signed [7:0] a,
  input real in_real,    // <-- Problematic: real type
  output real out_real   // <-- Problematic: real type
);
  logic cmp_result;

  always_comb begin
    cmp_result = (-a <= a) ? 1 : 0;
  end

  always @(posedge clk) begin
    out_real <= in_real;  // Assignment involving real type triggers the bug
  end
endmodule
```

The `real` type (64-bit IEEE floating-point in SystemVerilog) is not properly handled by `hw::getBitWidth()`, which returns -1 for unsupported types.

## Root Cause

**Primary Issue:** Missing validation in `Mem2Reg.cpp` line 1753

The code at `lib/Dialect/LLHD/Transforms/Mem2Reg.cpp:1752-1753`:
```cpp
auto type = getStoredType(slot);
auto flatType = builder.getIntegerType(hw::getBitWidth(type));
```

This code assumes `hw::getBitWidth(type)` always returns a valid positive bit width. However, when `type` is a floating-point type (or any other type not implementing `BitWidthTypeInterface` properly), `getBitWidth()` returns -1.

**Secondary Issue:** The `hw::getBitWidth()` function returns -1 for unsupported types, but callers do not check for this error condition before using the result.

## Affected Components

- **Dialect:** LLHD (Low-Level Hardware Description)
- **Pass:** Mem2RegPass (Memory to Register Promotion)
- **File:** `lib/Dialect/LLHD/Transforms/Mem2Reg.cpp`
- **Line:** 1753 (and potentially 1752)

## Triggering Conditions

1. SystemVerilog code with `real` (floating-point) type variables
2. The variable must be involved in a clocked assignment (`always @(posedge clk)`)
3. The Mem2Reg pass must attempt to promote the signal

## Suggested Fix

Add validation before creating the integer type:

```cpp
auto type = getStoredType(slot);
int64_t bitWidth = hw::getBitWidth(type);
if (bitWidth < 0) {
  // Handle unsupported type - either skip, emit error, or use appropriate handling
  return signalPassFailure();  // Or emit diagnostic
}
auto flatType = builder.getIntegerType(static_cast<unsigned>(bitWidth));
```

Alternatively, the pass could skip promotion of variables with unsupported types or emit a proper diagnostic instead of crashing.

## Severity

**High** - This is an assertion failure that causes a complete crash of the tool, preventing any further processing. The bug can be triggered by valid SystemVerilog code that uses floating-point (`real`) types.

## Reproducibility

**100% reproducible** with the provided test case using:
```bash
circt-verilog --ir-hw source.sv
```
