# Duplicate Issue Check Report

## Summary

| Metric | Value |
|--------|-------|
| **Recommendation** | `review_existing` |
| **Top Similarity Score** | 12.5 |
| **Most Similar Issue** | [#8332](https://github.com/llvm/circt/issues/8332) |

## Search Queries Used

1. `string port MooreToCore`
2. `StringType`
3. `MooreToCore type converter`
4. `dyn_cast non-existent`
5. `Moore string`

## Matching Issues

### 🔴 High Similarity (score >= 10.0) - Review Existing

#### Issue #8332: [MooreToCore] Support for StringType from moore to llvm dialect
- **Score**: 12.5
- **URL**: https://github.com/llvm/circt/issues/8332
- **Matched Keywords**:
  - `StringType` (title: 2.0)
  - `MooreToCore` (title: 2.0)
  - `string` (body: 1.0)
  - `type conversion` (body: 1.0)
  - `typeConverter` (body: 1.0)
  - `convertType` (body: 1.0)
- **Relevance**: **Directly discusses StringType support in MooreToCore.** Same root cause - missing StringType converter.

### 🟡 Medium Similarity (5.0 <= score < 10.0) - Likely New

#### Issue #8283: [ImportVerilog] Cannot compile forward declared string type
- **Score**: 9.0
- **URL**: https://github.com/llvm/circt/issues/8283
- **Matched Keywords**:
  - `string` (title: 2.0)
  - `MooreToCore` (body: 1.0)
  - `type conversion` (body: 1.0)
  - `StringType` (body: 1.0)
- **Relevance**: Reports string type compilation failure in MooreToCore. Same underlying issue - lack of string type conversion.

#### Issue #7535: [MooreToCore] VariableOp lowered failed
- **Score**: 6.5
- **URL**: https://github.com/llvm/circt/issues/7535
- **Matched Keywords**:
  - `MooreToCore` (title: 2.0)
  - `type conversion` (body: 1.0)
  - `dyn_cast` (body: 1.0)
  - `hw::InOutType` (body: 1.0)
- **Relevance**: Similar crash pattern in MooreToCore with type conversion failure. Different trigger (struct type vs string type).

#### Issue #8930: [MooreToCore] Crash with sqrt/floor
- **Score**: 6.0
- **URL**: https://github.com/llvm/circt/issues/8930
- **Matched Keywords**:
  - `MooreToCore` (title: 2.0)
  - `dyn_cast non-existent` (assertion: 3.0)
  - `type conversion` (body: 1.0)
- **Relevance**: **Same assertion message** 'dyn_cast on a non-existent value'. Different trigger (real type conversion).

### 🟢 Low Similarity (score < 5.0) - New Issue

#### Issue #8219: [ESI] Assertion: dyn_cast on a non-existent value
- **Score**: 4.0
- **URL**: https://github.com/llvm/circt/issues/8219
- **Relevance**: Same assertion but in ESI dialect, not Moore. Different root cause.

#### Issue #8292: [MooreToCore] Support for Unsized Array Type
- **Score**: 4.0
- **URL**: https://github.com/llvm/circt/issues/8292
- **Relevance**: Related feature request for type support in MooreToCore. Different type (unsized array).

## Conclusion

**This crash is highly likely to be a duplicate of Issue #8332** or closely related to Issue #8283.

Both existing issues describe the same fundamental problem:
- MooreToCore lacks a type converter for `moore::StringType`
- When a module uses string types, the type converter returns null
- This causes downstream assertion failures

### Recommendation

Before filing a new issue:
1. **Review Issue #8332** - Check if your specific crash scenario (string as module port) is covered
2. **Consider commenting** on #8332 or #8283 with your test case as additional evidence
3. If filing new, **reference both #8332 and #8283** as related issues

### Key Differences from Existing Issues

- **#8332** focuses on lowering StringType variables to LLVM
- **#8283** focuses on string variable declaration
- **Current crash** is specifically about **string-typed module ports** causing `getModulePortInfo()` to fail

If the module port scenario is not explicitly covered in existing issues, this could be filed as a new issue referencing the existing ones.
