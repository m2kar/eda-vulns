# [LLHD][Mem2Reg] Assertion failure due to invalid integer bitwidth when processing class-type variables

## Summary

The LLHD Mem2Reg pass crashes with an assertion failure when processing SystemVerilog code containing class-type variables used in sequential logic blocks. The pass incorrectly computes the bitwidth of class types, resulting in an integer type with 1,073,741,823 bits (0x3FFFFFFF), which exceeds MLIR's maximum supported bitwidth of 16,777,215 bits.

## Minimal Reproducer

```systemverilog
module m(input clk);
  class c; endclass
  c o;
  always @(posedge clk) o = new();
endmodule
```

## Reproduction Command

```bash
circt-verilog --ir-hw bug.sv
```

## Error Output

```
bug.sv:2:9: remark: Class builtin functions (needed for randomization, constraints, and covergroups) are not yet supported and will be dropped during lowering.
  class c; endclass
        ^
bug.sv:3:5: error: 'hw.bitcast' op result #0 must be Type wherein the bitwidth in hardware is known, but got '!llvm.ptr'
  c o;
    ^
bug.sv:3:5: note: see current operation: %10 = "hw.bitcast"(%9) : (i1073741823) -> !llvm.ptr
```

## Version Information

- **CIRCT Version**: firtool-1.139.0
- **LLVM Version**: 22.0.0git
- **Embedded Slang**: 9.1.0+0

## Syntax Validation

The test case is confirmed to be valid SystemVerilog:
- **Verilator 5.022**: ✅ Pass (0 errors, 0 warnings)
- **Slang 10.0.6**: ✅ Pass (0 errors, 0 warnings)

## Root Cause Analysis

### Crash Location

The crash occurs in the LLHD Mem2Reg transformation pass at:
- **File**: `lib/Dialect/LLHD/Transforms/Mem2Reg.cpp`
- **Function**: `Promoter::insertBlockArgs`
- **Line**: ~1742

### Call Chain

```
Mem2RegPass::runOnOperation()
  └─ Promoter::promote()
       └─ Promoter::insertBlockArgs()
            └─ mlir::IntegerType::get()  [FAILS: bitwidth > 16777215]
```

### Underlying Cause

1. **Unsupported Class Feature**: The compiler correctly issues a remark that "Class builtin functions are not yet supported and will be dropped during lowering."

2. **Type Mishandling**: When the Mem2Reg pass encounters a class-type variable (`c o;`) during block argument insertion:
   - It attempts to determine the bitwidth of the type
   - The class type's size is incorrectly computed as 1,073,741,823 bits (2^30 - 1)
   - This value exceeds MLIR's 16,777,215-bit limit for integer types
   - The `IntegerType::get()` call fails its verification invariants

3. **Missing Validation**: The Mem2Reg pass does not properly validate or handle class types before attempting to convert them to MLIR integer types. Although classes are marked as unsupported, the lowering process doesn't fully remove or properly handle all class-related constructs before this pass runs.

### Key Constructs Triggering the Bug

| Construct | Code | Role |
|-----------|------|------|
| Class declaration | `class c; endclass` | Minimal class definition |
| Class-type variable | `c o;` | Triggers type computation |
| Sequential logic | `always @(posedge clk)` | Context for Mem2Reg processing |
| Class instantiation | `o = new();` | Dynamic object creation |

## Relationship to Existing Issues

This issue is **related to but distinct from Issue #8693** ("Local signal does not dominate final drive"):

- **Similarity Score**: 67.25%
- **Common Aspects**: Both involve the LLHD Mem2Reg pass and result in assertion failures
- **Key Differences**:
  - Issue #8693: Focuses on dominance issues with local signals
  - This issue: Focuses on invalid integer bitwidth computation when handling class types
  - Different trigger conditions and root causes

This appears to be a distinct variant that should be tracked separately, though fixes to either issue may provide insights for the other.

## Suggested Fix

1. **Input Validation**: The Mem2Reg pass should check for unsupported types (like class types) and skip or gracefully handle them instead of attempting promotion.

2. **Type Guard**: Before calling `IntegerType::get()`, validate that the computed bitwidth is within MLIR's limits:
   ```cpp
   if (bitwidth > IntegerType::kMaxWidth) {
     // Emit diagnostic and skip, don't crash
   }
   ```

3. **Complete Lowering**: Ensure class-related constructs are fully lowered or removed before passes that cannot handle them.

4. **Error Emission**: Instead of crashing on assertion, emit a proper diagnostic error when encountering unsupported types during optimization passes.

## Additional Context

- **Severity**: Medium (crashes on unsupported but valid SystemVerilog)
- **Reproducibility**: Deterministic (100% reproducible)
- **Workaround**: Avoid using SystemVerilog classes in designs processed by circt-verilog

## Related Source Files

- `lib/Dialect/LLHD/Transforms/Mem2Reg.cpp`
- `llvm/mlir/lib/IR/MLIRContext.cpp`
- `llvm/mlir/include/mlir/IR/StorageUniquerSupport.h`
