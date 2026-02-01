# [LLHD] Crash when using `real` type in sequential logic blocks

## Description

CIRCT crashes with an internal error when processing SystemVerilog code that uses the `real` (floating-point) type in sequential logic blocks (e.g., `always_ff`). The crash manifests as an assertion failure or validation error involving an invalid bitwidth for hardware types.

While related to issues #8930 and #8269 concerning `real` type support in MooreToCore conversion, this issue represents a specific crash scenario demonstrating that `real` types in sequential logic blocks trigger an invalid state in the LLHD Mem2Reg transformation pass.

## Steps to Reproduce

1. Create a SystemVerilog file with a `real` type variable in a sequential logic block:

```systemverilog
module test(input logic clk);
  real r;
  always_ff @(posedge clk) r <= 1.0;
endmodule
```

2. Run the CIRCT verilog compiler:

```bash
circt-verilog --ir-hw test.sv
```

3. Observe the crash/error

## Expected Behavior

The compiler should either:
1. Successfully compile the valid SystemVerilog code (if `real` type support is implemented), or
2. Emit a clear and graceful "unsupported feature" diagnostic message

Instead of crashing with an internal assertion failure or creating an invalid MLIR intermediate representation.

## Actual Behavior

The compiler produces one of the following errors:

**Current Error (with improved validation):**
```
test.sv:2:8: error: 'hw.bitcast' op result #0 must be Type wherein the bitwidth in hardware is known, but got 'f64'
  real r;
       ^
note: see current operation: %11 = "hw.bitcast"(%10) : (i1073741823) -> f64
```

**Original Error (older versions with assertion failure):**
```
error: integer bitwidth is limited to 16777215 bits
Assertion `succeeded(ConcreteT::verifyInvariants(...))' failed.
```

The key issue is the invalid bitwidth `i1073741823` (= 2^30 - 1) being created during type conversion, which exceeds the hardware bitwidth limit of 16777215 (= 2^24 - 1).

## Root Cause

The LLHD Mem2Reg (Memory-to-Register Promotion) pass, invoked during the compilation pipeline, fails to properly handle floating-point types:

### Technical Details

1. **Type Conversion Failure**: When the SystemVerilog `real` type is parsed via the Moore dialect and converted to core dialects through MooreToCore, it becomes `Float64Type` (MLIR's 64-bit floating-point type).

2. **Bitwidth Calculation Issue**: The `hw::getBitWidth()` utility function in `lib/Dialect/HW/HWTypes.cpp` does not handle MLIR floating-point types properly. For `Float64Type`, it returns `-1` (unknown bitwidth):

   ```cpp
   return llvm::TypeSwitch<::mlir::Type, int64_t>(type)
       .Case<IntegerType>(
           [](IntegerType t) { return t.getIntOrFloatBitWidth(); })
       .Default([](Type type) -> int64_t {
           if (auto iface = dyn_cast<BitWidthTypeInterface>(type)) {
               std::optional<int64_t> width = iface.getBitWidth();
               return width.has_value() ? *width : -1;
           }
           return -1;  // <- Float64Type returns -1 here
       });
   ```

3. **Invalid IntegerType Creation**: The Mem2Reg pass's `Promoter::insertBlockArgs()` function (in `lib/Dialect/LLHD/Transforms/Mem2Reg.cpp:1742`) attempts to create block arguments for promoted memory/signal slots. When processing a slot with a floating-point type:
   - It retrieves the bitwidth using `hw::getBitWidth()`, which returns `-1`
   - This invalid value is used to create an `IntegerType`
   - When interpreted as unsigned, `-1` becomes `0xFFFFFFFF`, which after masking yields `1073741823` (2^30 - 1)
   - This causes either an assertion failure or a later validation error

4. **Missing Type Support**: The Mem2Reg pass was designed for integer/bit-vector types typical in hardware designs and does not properly validate or handle floating-point types before attempting promotion.

### Call Stack
```
Promoter::insertBlockArgs(BlockEntry*) [Mem2Reg.cpp:1742]
  ↑
Promoter::insertBlockArgs() [Mem2Reg.cpp:1654]
  ↑
Promoter::promote() [Mem2Reg.cpp:764]
  ↑
Mem2RegPass::runOnOperation() [Mem2Reg.cpp:1844]
```

## Environment

- **CIRCT Version**: firtool-1.139.0 (based on LLVM 22)
- **Toolchain**: llvm-22
- **SystemVerilog Standard**: IEEE 1800-2017 (contains `real` type)
- **Affected Passes**:
  - LLHD Mem2Reg pass (`lib/Dialect/LLHD/Transforms/Mem2Reg.cpp`)
  - MooreToCore conversion (`lib/Conversion/MooreToCore/MooreToCore.cpp`)
  - HW bitwidth utilities (`lib/Dialect/HW/HWTypes.cpp`)

## Validation

The test case has been validated as:
- ✅ **Syntactically Valid**: Passes Verilator (`verilator --lint-only`) and Slang linters
- ✅ **Reproducible**: Consistently crashes on CIRCT
- ✅ **Valid Bug Report**: Other synthesis tools accept the code without errors

This confirms the issue is a genuine compiler bug, not an invalid test case or unsupported language feature.

## Related Issues

This issue is related to but distinct from:

- **#8930** - "[MooreToCore] Crash with sqrt/floor"
  - Similarity: 8/10 - Same root cause (real type handling in MooreToCore)
  - Difference: That issue involves real-type function calls; this involves real in sequential logic
  - Both stem from `hw::getBitWidth()` failing for `Float64Type`

- **#8269** - "[MooreToCore] Support `real` constants"
  - Similarity: 8/10 - Directly related to `real` type support gap
  - Difference: That issue is about constant handling; this is about variable storage
  - Both indicate incomplete `real` type support in CIRCT

**Note**: The duplicate check analysis assigned a similarity score of **8/10** for #8930 and #8269, which is below the 10.0 threshold for marking as an exact duplicate. This crash represents a specific manifestation in sequential logic processing that merits separate tracking while acknowledging the related root cause.

## Suggested Fixes

### Short-term (Defensive)
Add type validation in the Mem2Reg pass to skip or explicitly reject non-promotable types (floating-point, strings, etc.) before attempting promotion:

```cpp
// In Promoter::insertBlockArgs()
if (isa<FloatType>(slotType)) {
    emitError(loc) << "Cannot promote floating-point type signals; "
                   << "real type is not supported in memory-to-register promotion";
    return failure();
}
```

### Long-term (Proper Support)
1. **Extend `hw::getBitWidth()`** to properly handle MLIR floating-point types instead of returning `-1`
2. **Add type conversion rules** in MooreToCore for proper `Float64Type` handling
3. **Implement floating-point register support** in LLHD dialect if semantically appropriate, or
4. **Document explicit limitations** and emit clear errors early in the compilation pipeline

## Additional Context

- **Crash ID**: 260129-0000193c
- **Type of Crash**: Assertion failure / Validation error
- **Severity**: High (compiler crash)
- **Reproducibility**: Deterministic (always crashes on valid `real` type input)
- **Scope**: Affects any SystemVerilog design using floating-point types in sequential logic blocks

The minimal test case (`bug.sv`) demonstrates the simplest possible reproduction:
```systemverilog
module test(input logic clk);
  real r;
  always_ff @(posedge clk) r <= 1.0;
endmodule
```

This issue is critical because:
1. It causes compiler crashes on valid SystemVerilog syntax
2. It produces confusing error messages with impossible bitwidth values
3. It blocks compilation of testbenches and behavioral models using `real` types
4. Users expect either proper support or a clear "unsupported" message, not a crash
