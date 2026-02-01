# [Moore] circt-verilog --ir-hw hangs on always_comb with bit-select assignment

## Description

`circt-verilog --ir-hw` hangs indefinitely (timeout) when processing a simple `always_comb` block with bit-select assignment. The compilation appears to enter an infinite loop or exponential complexity state during the Moore to Core conversion.

## Minimal Testcase

```systemverilog
module top;
  logic [7:0] data;
  always_comb data[0] = ~data[7];
endmodule
```

## Reproduction Steps

```bash
# Save the testcase to bug.sv and run:
timeout 30s circt-verilog --ir-hw bug.sv
# Expected: Exit code 124 (timeout after 30 seconds)
```

## Expected Behavior

The compilation should complete successfully (similar to verilator and slang which accept this code).

## Actual Behavior

`circt-verilog --ir-hw` hangs indefinitely. The process does not terminate and must be killed with a timeout signal.

## Toolchain Information

- **CIRCT Version**: 1.139.0
- **circt-verilog**: `/opt/firtool/bin/circt-verilog`
- **LLVM Version**: 22.0.0git

## Cross-Tool Validation

| Tool | Command | Result |
|------|---------|--------|
| **Verilator** | `verilator --lint-only bug.sv` | ✅ Pass (exit code 0) |
| **Slang** | `slang --lint-only bug.sv` | ✅ Pass (exit code 0) |
| **CIRCT Parse** | `circt-verilog --parse-only bug.sv` | ✅ Pass (exit code 0) |
| **CIRCT IR (Moore)** | `circt-verilog --ir-moore bug.sv` | ✅ Pass (exit code 0) |
| **CIRCT IR (HW)** | `circt-verilog --ir-hw bug.sv` | ❌ Timeout (exit code 124) |

## Root Cause Analysis

### Affected Stage
The timeout occurs during the **Moore to Core conversion** (triggered by `--ir-hw` flag). The parsing and Moore IR generation stages complete successfully.

### Trigger Conditions
The timeout is triggered by the specific combination of:
1. An `always_comb` block
2. Bit-select assignment (`data[0]` and `data[7]`)
3. Both source and target bits from the same signal
4. Signal width ≥ 8 bits

### Suspected Location
Based on the minimal testcase and affected stage, the issue likely resides in:
- **Primary**: `lib/Conversion/MooreToCore/MooreToCore.cpp` - Always_comb processing
- **Specific**: Value observation logic (`getValuesToObserve`) or combinatorial logic conversion

The hang may be caused by:
- Infinite loop in value dependency analysis
- Exponential complexity in fan-in/cone analysis for self-referencing bit operations
- Incorrect handling of bit-select operations in combinatorial logic conversion

## Additional Findings

During minimization, a **separate bug** was discovered:

```systemverilog
module top;
  logic [1:0] data;
  always_comb data[0] = ~data[1];
endmodule
```

This smaller testcase causes a **segmentation fault** (exit code 139) in `circt::comb::XorOp::canonicalize` instead of a timeout. This appears to be a different issue but related to the same area of code (bit-select operations in `always_comb`).

## Related Issues

The following issues have been reviewed and determined to be unrelated:

- **#9570** - MooreToCore assertion with packed union types (different trigger)
- **#8844** - 'moore.case_eq' type error (different error type)
- **#8176** - MooreToCore crash with unattached region (different crash type)
- **#8211** - Unexpected observed values (different symptom)

This issue is unique due to:
- **Timeout** behavior (vs assertion/segfault in related issues)
- **Bit-select assignment** in `always_comb` as the minimal trigger
- **No nested modules or functions** required to reproduce

## Original Testcase (Pre-Minimization)

The original fuzzer-generated testcase was more complex with nested modules and function chains, but minimization revealed that these constructs are not necessary to trigger the bug. The simplified testcase above represents the core issue.

**Original**: source.sv (451 bytes)  
**Minimized**: bug.sv (200 bytes) - 55.7% reduction

## Severity

**High** - This bug can cause the compiler to hang indefinitely on valid SystemVerilog code, making it difficult to diagnose issues in real designs that use bit-select operations in combinatorial logic.

## Suggested Fix

Review the value observation and dependency analysis logic in the MooreToCore conversion, specifically for bit-select operations within `always_comb` blocks. Ensure that self-referencing bit assignments do not cause infinite loops or exponential complexity.

## Labels

- [Moore]
- [circt-verilog]
- [MooreToCore]
- bug
- timeout
- always_comb
