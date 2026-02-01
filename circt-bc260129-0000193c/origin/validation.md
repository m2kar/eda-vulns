# Validation Report

## Summary

| Check | Result |
|-------|--------|
| Syntax Validity | ✅ PASS |
| Bug Reproduction | ✅ PASS |
| Cross-Tool Validation | ✅ PASS |
| Classification | **REPORT** |

## Test Case

```systemverilog
// Minimized test case for CIRCT bug: real type in sequential logic
// Original crash: integer bitwidth is limited to 16777215 bits
// Actual error: hw.bitcast with invalid bitwidth (i1073741823 -> f64)
module test(input logic clk);
  real r;
  always_ff @(posedge clk) r <= 1.0;
endmodule
```

## Syntax Validation

### Verilator
```
Command: verilator --lint-only bug.sv
Exit Code: 0
Result: PASS (no errors, no warnings)
```

### Slang
```
Command: slang --lint-only bug.sv
Exit Code: 0
Result: PASS
Output: Build succeeded: 0 errors, 0 warnings
```

## Bug Reproduction

### CIRCT (circt-verilog)
```
Command: circt-verilog --ir-hw bug.sv
Version: firtool-1.139.0
Exit Code: 1
Result: BUG REPRODUCED
```

**Error Output:**
```
origin/bug.sv:5:8: error: 'hw.bitcast' op result #0 must be Type wherein the bitwidth in hardware is known, but got 'f64'
  real r;
       ^
origin/bug.sv:5:8: note: see current operation: %11 = "hw.bitcast"(%10) : (i1073741823) -> f64
```

## Analysis

### Error Interpretation

The error reveals the underlying bug:
- **Invalid bitwidth**: `i1073741823` (= 2^30 - 1)
- **Expected type**: `f64` (Float64Type from SystemVerilog `real`)
- **Root cause**: `hw::getBitWidth()` returns -1 for Float64Type, which is incorrectly used

### Bitwidth Calculation

```
-1 (signed) = 0xFFFFFFFF (unsigned 32-bit) & 0x3FFFFFFF (mask) = 1073741823
```

This explains why the original crash message mentioned "16777215 bits" (2^24 - 1) in some contexts, while the current manifestation shows 1073741823 (2^30 - 1).

## Cross-Tool Validation Summary

| Tool | Syntax Valid | Compiles Successfully |
|------|--------------|----------------------|
| Verilator | ✅ Yes | ✅ Yes (lint-only) |
| Slang | ✅ Yes | ✅ Yes |
| CIRCT | ✅ Yes | ❌ No (Bug) |

**Conclusion**: The test case uses valid SystemVerilog syntax and is accepted by industry-standard tools. The failure is specific to CIRCT, confirming this is a compiler bug.

## Feature Analysis

The `real` type in SystemVerilog:
- Is part of IEEE 1800-2017 standard
- Represents 64-bit floating-point numbers
- Commonly used for modeling and testbenches
- Hardware synthesis tools may not support it (as it's not synthesizable)

**However**, CIRCT should:
1. Either support the `real` type properly in its flow
2. Or emit a clear "unsupported feature" error message

Instead, it crashes/errors with an invalid internal state (impossible bitwidth).

## Validation Result

| Attribute | Value |
|-----------|-------|
| **Classification** | `report` |
| **Valid Test Case** | Yes |
| **Bug Reproducible** | Yes |
| **Confidence** | High |
| **Severity** | High (compiler crash/internal error) |

## Recommendation

This is a valid bug report that should be filed against CIRCT. The issue is:

1. **Primary Bug**: LLHD Mem2Reg pass (or type conversion) fails to handle `Float64Type`
2. **Symptom**: Invalid bitwidth error (1073741823 or assertion failure)
3. **Impact**: Cannot process valid SystemVerilog with `real` type in sequential logic

The fix should either:
- Add proper Float64Type support in the affected passes
- Skip promotion of non-integer types in Mem2Reg
- Emit a clear "unsupported" diagnostic instead of crashing
