# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Searched | 6 |
| Top Similarity Score | **9.5** |
| **Recommendation** | **review_existing** ⚠️ |

## Search Parameters

- **Dialect**: Moore
- **Failing Pass**: MooreToCore
- **Crash Type**: assertion
- **Assertion Message**: `dyn_cast on a non-existent value`
- **Keywords**: string, port, MooreToCore, getModulePortInfo, dyn_cast, InOutType, type conversion, assertion, sanitizeInOut, DynamicStringType

## Top Similar Issues

### #1 [#8930](https://github.com/llvm/circt/issues/8930) (Score: 9.5) ⚠️ HIGH

**Title**: [MooreToCore] Crash with sqrt/floor

**State**: OPEN

**Match Reasons**:
- Title match: 'MooreToCore' (+2.0)
- Body match: 'MooreToCore' (+1.0)
- Body match: 'dyn_cast' (+1.0)
- Body match: 'assertion' (+1.0)
- **Assertion match: 'dyn_cast on a non-existent value' (+3.0)**

**Analysis**: This issue has the **exact same assertion message** (`dyn_cast on a non-existent value`) in MooreToCore pass. However:
- **#8930 crash location**: `ConversionOpConversion::matchAndRewrite` (real type conversion with sqrt/floor)
- **Our crash location**: `getModulePortInfo` (string type as module port)

**Verdict**: Same assertion, different root cause. These are **related but distinct bugs**.

---

### #2 [#8283](https://github.com/llvm/circt/issues/8283) (Score: 7.0)

**Title**: [ImportVerilog] Cannot compile forward declared string type

**State**: OPEN

**Match Reasons**:
- Title match: 'string' (+2.0)
- Body match: 'string' (+1.0)
- Body match: 'MooreToCore' (+1.0)
- Body match: 'type conversion' (+1.0)

**Analysis**: Related to string type handling in MooreToCore, but different error (legalization failure vs assertion crash).

---

### #3 [#8332](https://github.com/llvm/circt/issues/8332) (Score: 7.0)

**Title**: [MooreToCore] Support for StringType from moore to llvm dialect

**State**: OPEN

**Match Reasons**:
- Title match: 'string' (+2.0)
- Title match: 'MooreToCore' (+2.0)
- Body match: 'string' (+1.0)

**Analysis**: Feature request for string type support, not a crash report.

---

### #4 [#9206](https://github.com/llvm/circt/issues/9206) (Score: 6.5)

**Title**: [ImportVerilog] moore.conversion generated instead of moore.int_to_string

**State**: OPEN

**Analysis**: Related to string operations but different issue (wrong op selection).

---

### #5 [#7629](https://github.com/llvm/circt/issues/7629) (Score: 5.5)

**Title**: [MooreToCore] Support net op

**State**: OPEN

**Analysis**: Unrelated - about net operation support.

---

## Recommendation

**Action**: `review_existing` ⚠️

### Key Finding

**Issue #8930** shares the **exact same assertion message**:
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

However, the crash locations are different:
- **#8930**: `ConversionOpConversion::matchAndRewrite` in MooreToCore.cpp (handling `moore.conversion` for real types)
- **Our bug**: `getModulePortInfo` in MooreToCore.cpp (handling string type as module output port)

### Conclusion

**Not a duplicate, but related.** Both issues expose the same underlying problem: MooreToCore's type converter returns empty/invalid types that are then passed to `dyn_cast` without validation.

### Recommended Action

1. **Create a new issue** - the crash location and trigger are different
2. **Reference #8930** in the new issue as a related bug
3. **Suggest a common fix**: Add null checks after `typeConverter.convertType()` calls

## Scoring Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
