# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 3 |
| Top Similarity Score | 12.5 |
| **Recommendation** | **review_existing** |

## Search Parameters

- **Dialect**: Moore
- **Failing Pass**: MooreToCore
- **Crash Type**: assertion
- **Keywords**: string, StringType, sanitizeInOut, dyn_cast, port type, MooreToCore

## Top Similar Issues

### [#9572](https://github.com/llvm/circt/issues/9572) (Score: 12.5)

**Title**: [Moore] Assertion failure when module has string type output port

**State**: OPEN

---

### [#9570](https://github.com/llvm/circt/issues/9570) (Score: 7.0)

**Title**: [Moore] Assertion in MooreToCore when module uses packed union type as port

**State**: OPEN

---

### [#8930](https://github.com/llvm/circt/issues/8930) (Score: 4.0)

**Title**: [MooreToCore] Crash with sqrt/floor

**State**: OPEN

---

## Recommendation

**Action**: `review_existing`

⚠️ **High similarity detected**. Issue #9572 appears to describe the same crash scenario (string-typed module ports causing a MooreToCore assertion). Review and, if it matches, add this minimized test case and crash log to that issue instead of creating a new one.
