# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| **Recommendation** | `new_issue` |
| **Top Similarity Score** | 0.72 |
| **Most Similar Issue** | #8930 |
| **Total Issues Searched** | 25 |

## Search Keywords

- `UnionType`, `packed union`, `MooreToCore`
- `type conversion`, `module port`
- `dyn_cast on a non-existent value`

## Analysis Result

**This appears to be a NEW issue** - No existing issue specifically reports missing `moore::UnionType` converter in MooreToCore pass.

### Most Similar Issue: #8930

**Title**: [MooreToCore] Crash with sqrt/floor

**Similarity**: 72%

**Comparison**:
| Aspect | #8930 | This Bug |
|--------|-------|----------|
| Pass | MooreToCore | MooreToCore |
| Assertion | `dyn_cast on a non-existent value` | `dyn_cast on a non-existent value` |
| Trigger | `moore.conversion` (real type) | `SVModuleOp` (union port type) |
| Crash Location | `ConversionOpConversion` | `getModulePortInfo` |
| Root Cause | Missing real→integer conversion | Missing UnionType converter |

**Verdict**: Same assertion message but **different root cause**. #8930 is about `real` type conversion, this bug is about `UnionType` in module ports.

### Other Related Issues

| Issue | Title | Score | Relevance |
|-------|-------|-------|-----------|
| #8471 | [ImportVerilog] Union type in call | 0.45 | Union type, different error |
| #7535 | [MooreToCore] VariableOp lowered failed | 0.40 | MooreToCore, struct type issue |
| #7378 | [HW] Roundtrip test fail for !hw.union | 0.35 | HW union parsing |
| #967 | Union type | 0.30 | Historical feature discussion |

## Recommendation

**Proceed with new issue submission.**

The bug represents a distinct missing feature: MooreToCore lacks a type converter for `moore::UnionType`. While #8930 shares the same assertion message, it's triggered by a completely different code path (real type conversion vs module port type conversion).

### Suggested Cross-References

When filing the issue, consider mentioning:
- #8930 - Similar assertion, different root cause
- #7535 - Related MooreToCore type conversion pattern
