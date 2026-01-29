# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| **Recommendation** | `review_existing` |
| **Top Similarity Score** | 12.5 |
| **Top Matching Issue** | #8283 |
| **Total Issues Found** | 7 |

## Recommendation

**⚠️ REVIEW EXISTING ISSUES BEFORE FILING**

The current crash (string type as module output port causing assertion failure in `getModulePortInfo()`) appears to be closely related to existing issues about string type support in MooreToCore.

## Top Matching Issues

### 1. Issue #8283 - [ImportVerilog] Cannot compile forward declared string type
- **Score**: 12.5 (HIGH)
- **State**: Open
- **URL**: https://github.com/llvm/circt/issues/8283
- **Relevance**: Directly reports that MooreToCore lacks string-type conversion. The issue states: "This error is due to MooreToCore's lack of string-type conversion."

**Key Quote from Issue:**
> `circt-verilog` has the following complaints when compiling the string.sv file:
> `error: failed to legalize operation 'moore.variable'`
> This error is due to MooreToCore's lack of string-type conversion.

### 2. Issue #8332 - [MooreToCore] Support for StringType from moore to llvm dialect
- **Score**: 11.0 (HIGH)
- **State**: Open
- **URL**: https://github.com/llvm/circt/issues/8332
- **Relevance**: Feature request for StringType support in MooreToCore. Discusses implementation approach for string types.

### 3. Issue #8930 - [MooreToCore] Crash with sqrt/floor
- **Score**: 9.5 (MEDIUM-HIGH)
- **State**: Open
- **URL**: https://github.com/llvm/circt/issues/8930
- **Relevance**: **Same assertion message**: `Assertion 'detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed`. Different trigger (real type conversion) but identical crash pattern.

### 4. Issue #8173 - [ImportVerilog] Crash on ordering-methods-reverse test
- **Score**: 7.0 (MEDIUM)
- **State**: Open
- **URL**: https://github.com/llvm/circt/issues/8173
- **Relevance**: Crash involving string type arrays. Different crash location but related to string handling.

## Comparison with Current Bug

| Aspect | Current Bug | #8283 | #8930 |
|--------|-------------|-------|-------|
| **Crash Location** | `getModulePortInfo()` | `moore.variable` legalization | `ConversionOpConversion` |
| **Assertion** | `dyn_cast on a non-existent value` | N/A (legalization error) | `dyn_cast on a non-existent value` |
| **Type Involved** | `string` (output port) | `string` (variable) | `real` (conversion) |
| **Root Cause** | StringType → DynamicStringType not valid for hw::PortInfo | Missing string-type conversion | Missing real type handling |

## Analysis

The current bug is a **specific manifestation** of the broader string type support gap in MooreToCore:

1. **#8283** reports the general problem: MooreToCore cannot handle string types
2. **Current bug** shows a specific crash path: string type used as module output port
3. **#8930** shows the same assertion pattern with a different unsupported type

### Recommendation Actions

1. **Comment on #8283** with this specific crash case as additional evidence
2. **Reference #8930** as showing the same assertion pattern
3. **Consider filing as new issue** only if:
   - The port-specific crash path is considered distinct enough
   - Maintainers prefer separate tracking for different crash scenarios

## Search Queries Used

1. `string port MooreToCore`
2. `getModulePortInfo assertion`
3. `Moore string type`
4. `dyn_cast InOutType`
5. `MooreToCore crash`
6. `DynamicStringType`
7. `SVModuleOp port`
8. `Moore type conversion crash`
9. `output port type`
10. `PortInfo assertion`

## All Matching Issues

| # | Issue | Title | Score | State |
|---|-------|-------|-------|-------|
| 1 | #8283 | [ImportVerilog] Cannot compile forward declared string type | 12.5 | Open |
| 2 | #8332 | [MooreToCore] Support for StringType from moore to llvm dialect | 11.0 | Open |
| 3 | #8930 | [MooreToCore] Crash with sqrt/floor | 9.5 | Open |
| 4 | #8173 | [ImportVerilog] Crash on ordering-methods-reverse test | 7.0 | Open |
| 5 | #8292 | [MooreToCore] Support for Unsized Array Type | 5.0 | Open |
| 6 | #8176 | [MooreToCore] Crash when getting values to observe | 4.5 | Open |
| 7 | #8211 | [MooreToCore] Unexpected observed values in llhd.wait | 4.0 | Open |
