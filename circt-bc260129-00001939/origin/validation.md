# Validation Report

## Classification

| Field | Value |
|-------|-------|
| **Result** | `report` |
| **Reason** | Valid SystemVerilog code causes an assertion failure in CIRCT's Mem2Reg pass |
| **Reduction** | 83.3% (18 → 3 lines) |

## Syntax Validation

### Verilator (v5.022)
```
$ verilator --lint-only bug.sv
(no output - passed)
```
**Result**: ✅ PASS

### Slang (v10.0.6)
```
$ slang bug.sv
Top level design units:
    m

Build succeeded: 0 errors, 0 warnings
```
**Result**: ✅ PASS

## Crash Signature Verification

### Error Message Match
- **Original**: `<unknown>:0: error: integer bitwidth is limited to 16777215 bits`
- **Minimized**: `<unknown>:0: error: integer bitwidth is limited to 16777215 bits`
- **Match**: ✅ Identical

### Assertion Match
- **Original**: `Assertion 'succeeded(ConcreteT::verifyInvariants(...))' failed`
- **Minimized**: `Assertion 'succeeded(ConcreteT::verifyInvariants(...))' failed`
- **Match**: ✅ Identical

### Crash Location Match
- **Original**: `Mem2Reg.cpp:1742` in `Promoter::insertBlockArgs`
- **Minimized**: `Mem2Reg.cpp:1742` in `Promoter::insertBlockArgs`
- **Match**: ✅ Identical

## Minimized Test Case

```systemverilog
module m(input c, output real o);
always @(posedge c) o <= 0;
endmodule
```

## Reproduction Command

```bash
/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw bug.sv
```

## Bug Analysis

### Root Cause
The LLHD Mem2Reg pass calls `hw::getBitWidth()` on the stored type of a memory slot. For floating-point types (`real`/`f64`), this returns an invalid value (-1 or a very large value). This invalid bitwidth is then passed to `builder.getIntegerType()`, which triggers an assertion because MLIR limits integer bitwidth to 16,777,215 bits.

### Minimal Trigger Conditions
1. Module with `real` type output port
2. Clocked `always` block (creates sequential logic)
3. Non-blocking assignment to the real output

### Why This Is a Bug
- The test case is valid SystemVerilog (confirmed by Verilator and Slang)
- CIRCT should either:
  - Properly handle `real` types in sequential logic, OR
  - Emit a meaningful diagnostic if `real` types are not supported in this context
- Instead, it crashes with an assertion failure

## Conclusion

**This is a reportable bug.** The test case is syntactically valid SystemVerilog that causes CIRCT to crash due to improper handling of floating-point types in the Mem2Reg optimization pass.
