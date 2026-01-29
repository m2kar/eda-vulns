# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 10 |
| Top Similarity Score | 3.0 |
| **Recommendation** | **new_issue** |

## Search Parameters

- **Dialect**: Moore
- **Failing Pass**: MooreToCore
- **Crash Type**: assertion
- **Keywords**: string port, DynamicStringType, MooreToCore, module port, simulation type, HW dialect, type conversion, SystemVerilog string, port type, crash

## Top Similar Issues

### [#8283](https://github.com/llvm/circt/issues/8283) (Score: 3.0)

**Title**: [ImportVerilog] Cannot compile forward declared string type

**State**: OPEN

**Labels**: (none)

---

### [#8332](https://github.com/llvm/circt/issues/8332) (Score: 3.0)

**Title**: [MooreToCore] Support for StringType from moore to llvm dialect

**State**: OPEN

**Labels**: (none)

---

### [#5640](https://github.com/llvm/circt/issues/5640) (Score: 2.0)

**Title**: [SV] Introduce SystemVerilog `string` type

**State**: OPEN

**Labels**: Verilog/SystemVerilog

---

### [#8292](https://github.com/llvm/circt/issues/8292) (Score: 2.0)

**Title**: [MooreToCore] Support for Unsized Array Type

**State**: OPEN

**Labels**: (none)

---

### [#9206](https://github.com/llvm/circt/issues/9206) (Score: 2.0)

**Title**: [ImportVerilog] moore.conversion generated instead of moore.int_to_string

**State**: OPEN

**Labels**: Moore, ImportVerilog

---

### [#9490](https://github.com/llvm/circt/issues/9490) (Score: 1.5)

**Title**: [OM][Evaluator][FIRRTL] Should we be able to evaluate ext classes if we don't use outputs?

**State**: OPEN

**Labels**: FIRRTL, OM

---

### [#8930](https://github.com/llvm/circt/issues/8930) (Score: 1.0)

**Title**: [MooreToCore] Crash with sqrt/floor

**State**: OPEN

**Labels**: Moore

---

### [#8176](https://github.com/llvm/circt/issues/8176) (Score: 1.0)

**Title**: [MooreToCore] Crash when getting values to observe

**State**: OPEN

**Labels**: Moore

---

### [#7531](https://github.com/llvm/circt/issues/7531) (Score: 1.0)

**Title**: [Moore] Input triggers assertion in canonicalizer infra

**State**: OPEN

**Labels**: bug, Moore

---

### [#8211](https://github.com/llvm/circt/issues/8211) (Score: 1.0)

**Title**: [MooreToCore]Unexpected observed values in llhd.wait.

**State**: OPEN

**Labels**: Moore

---

## Recommendation

**Action**: `new_issue`

✅ **Clear to Proceed**

No similar issues were found. This is likely a new bug.

**Recommended**:
- Proceed to generate and submit bug report
- Reference related issues in report
- Highlight what makes this bug different

## Analysis of Similar Issues

### Most Similar: #8283 - Cannot compile forward declared string type

**Similarity Score**: 3.0

**Why it's similar**:
- Both involve MooreToCore and string type
- Both discuss string type conversion issues in CIRCT

**Why it's different**:
- Issue #8283 is about forward declarations of string types
- This issue is about string types used as **module output ports** causing crashes
- The crash signature is different (compilation error vs. segfault)
- The root cause is different ( legalization failure vs. type mismatch in hardware module)

### Related: #8332 - Support for StringType from moore to llvm dialect

**Similarity Score**: 3.0

**Why it's related**:
- Both discuss string type support in MooreToCore
- Both involve the lack of proper string type conversion

**Why it's different**:
- Issue #8332 is a feature request for supporting StringType
- This issue is a bug report for a crash
- This bug demonstrates a crash scenario that needs immediate fixing

### Other Related Issues

Several other issues discuss string type support and MooreToCore crashes, but none specifically address:
- String types as module **output ports**
- The specific crash in `SVModuleOpConversion::matchAndRewrite` when processing string ports
- The conversion of StringType → sim::DynamicStringType causing hardware module incompatibility

## Scoring Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
| Failing pass match | 2.0 | If failing pass appears in issue |

## Conclusion

The highest similarity score is 3.0 (issue #8283), which is below the threshold of 4.0 for "likely new" and 8.0 for "review existing". 

This bug is **unique** and should be submitted as a new GitHub issue. While there are related discussions about string type support in CIRCT, none specifically address the crash when string types are used as module output ports.

Key differentiators:
1. **Crash location**: `SVModuleOpConversion::matchAndRewrite` (MooreToCore)
2. **Crash trigger**: String type as module output port
3. **Crash type**: Segmentation fault (SIGSEGV) vs. compilation error
4. **Root cause**: sim::DynamicStringType incompatibility with HW module ports
