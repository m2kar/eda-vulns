# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 0 |
| Top Similarity Score | 0.0 |
| **Recommendation** | **new_issue** |

## Search Parameters

- **Dialect**: LLHD
- **Failing Pass**: Mem2Reg
- **Crash Type**: assertion
- **Keywords**: real type, floating point, Mem2Reg, LLHD, bitwidth, IntegerType, f64, Float64Type, RefType, getBitWidth, assertion, 16777215
- **Assertion Message**: "integer bitwidth is limited to 16777215 bits"

## Top Similar Issues

*No highly similar issues found.*

## Related Issues Found

While no exact duplicates were found, several related issues discuss similar topics:

### Related to Real Type Support
- **[#9234](https://github.com/llvm/circt/issues/9234): [ImportVerilog] Functionality for real number format specifiers not defined
- **[#8269](https://github.com/llvm/circt/issues/8269): [MooreToCore] Support `real` constants
- **[#2667](https://github.com/llvm/circt/issues/2667): Support of IEEE single and double precision floating point operations

### Related to Bitwidth Handling
- **[#9287](https://github.com/llvm/circt/issues/9287): [HW] Make `hw::getBitWidth` use std::optional vs -1
- **[#2593](https://github.com/llvm/circt/issues/2593): [ExportVerilog] Omit bitwidth of constant array index
- **[#6740](https://github.com/llvm/circt/issues/6740): [FIRRTLToHW] Conversion failure of uninstantiated module can crash LowerToHW

### Related to Mem2Reg/LLHD
- **[#8693](https://github.com/llvm/circt/issues/8693): [Mem2Reg] Local signal does not dominate final drive
- **[#9013](https://github.com/llvm/circt/issues/9013): [circt-opt] Segmentation fault during XOR op building

## Recommendation

**Action**: `new_issue`

✅ **Clear to Proceed**

No similar issues were found. This appears to be a new bug.

**Analysis**: While there are related issues about:
1. Limited support for `real` (floating-point) types in CIRCT
2. Bitwidth calculation issues with `hw::getBitWidth()`
3. Various crashes in the LLHD and Mem2Reg passes

None of these issues describe the specific bug where:
- SystemVerilog `real` type is used in `always_ff` (sequential) blocks
- The LLHD Mem2Reg pass crashes with the assertion "integer bitwidth is limited to 16777215 bits"
- The crash occurs because `hw::getBitWidth()` returns an invalid sentinel value (0x40000000 = 1,073,741,823 bits) for floating-point types

This is a distinct bug that should be reported separately.

## Scoring Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
| Failing pass match | 2.0 | If failing pass appears in issue |

## Next Steps

1. ✅ Proceed to generate the bug report
2. ✅ Reference the related issues (#9287, #8269, #2667) in the bug report
3. ✅ Submit the issue to llvm/circt repository
4. ⚠️ **DO NOT SUBMIT** - User requested not to submit, only generate report
