# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| **Recommendation** | `likely_new` - Likely a new issue |
| **Top Score** | 12.5 / 20 |
| **Top Issue** | #8283 |
| **Related Issues Found** | 4 |
| **Exact Duplicates** | 0 |

## Crash Signature

```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed
Location: getModulePortInfo() in MooreToCore.cpp:259
```

## Search Keywords

- `string` (primary)
- `port` (primary)
- `MooreToCore` (primary)
- `getModulePortInfo` (specific)
- `DynamicStringType` (specific)

## Related Issues

### #8283 - [ImportVerilog] Cannot compile forward decleared string type
| Attribute | Value |
|-----------|-------|
| **State** | 🟢 Open |
| **Score** | 12.5 / 20 |
| **URL** | https://github.com/llvm/circt/issues/8283 |

**Matching Keywords**: string, MooreToCore, moore.variable, string type

**Analysis**: 
Related issue about string type support in MooreToCore. Different crash location (`moore.variable` vs `getModulePortInfo`). This issue is about string **variables**, not string **ports** specifically. The error in #8283 is "failed to legalize operation 'moore.variable'", while our crash is an assertion failure in port info extraction.

**Verdict**: ⚠️ Possibly Related - Same root cause (incomplete string type support) but different manifestation

---

### #8332 - [MooreToCore] Support for StringType from moore to llvm dialect
| Attribute | Value |
|-----------|-------|
| **State** | 🟢 Open |
| **Score** | 11.0 / 20 |
| **URL** | https://github.com/llvm/circt/issues/8332 |

**Matching Keywords**: MooreToCore, StringType, string, sim dialect

**Analysis**: 
Feature request/discussion about lowering StringType to LLVM dialect. Not a crash report - it's a design discussion. No mention of port-specific issues or assertion failures.

**Verdict**: ℹ️ Related Context - Useful background but not a duplicate

---

### #7628 - [MooreToCore] Support string constants (Closed)
| Attribute | Value |
|-----------|-------|
| **State** | ⚫ Closed |
| **Score** | 7.0 / 20 |
| **URL** | https://github.com/llvm/circt/issues/7628 |

**Matching Keywords**: MooreToCore, string

**Analysis**: 
Closed issue about string constants support. String constants are different from dynamic string type used in ports.

**Verdict**: ✅ Not Duplicate - Different feature (constants vs port types)

---

### #8292 - [MooreToCore] Support for Unsized Array Type
| Attribute | Value |
|-----------|-------|
| **State** | 🟢 Open |
| **Score** | 3.0 / 20 |
| **URL** | https://github.com/llvm/circt/issues/8292 |

**Matching Keywords**: MooreToCore

**Analysis**: 
About unsized array types in MooreToCore. Tangentially related through the pass name but not about string types.

**Verdict**: ✅ Not Related

---

## Recommendation

### `likely_new` - This is Likely a New Issue

**Reasoning**:
1. **No exact duplicate found** - No existing issue reports a crash in `getModulePortInfo()` when processing string-typed ports
2. **Unique crash path** - The assertion `dyn_cast on a non-existent value` in port info extraction is not mentioned in any existing issue
3. **Different scenario from #8283** - While #8283 covers string variables, our bug is specifically about string type used as **module port**, which triggers a different code path
4. **Assertion vs Legalization failure** - #8283 shows a legalization error, while our crash is an assertion failure earlier in the conversion process

### Suggested Actions

1. **File as new issue** with reference to #8283 as related
2. **Cross-reference** the ongoing string type discussions (#8332)
3. **Highlight** that this is a crash/assertion failure, not just unsupported feature error

### Issue Title Suggestion

```
[MooreToCore] Assertion failure in getModulePortInfo when module has string-typed ports
```

## Score Breakdown Legend

| Component | Max Score | Description |
|-----------|-----------|-------------|
| Title Keywords | 4.0 | Matching keywords in issue title (weight 2.0 each) |
| Body Keywords | 6.0 | Matching keywords in issue body (weight 1.0 each) |
| Assertion Match | 6.0 | Exact assertion message match (weight 3.0) |
| Dialect Match | 1.5 | Same dialect/pass mentioned |
| Context Bonus | 2.5 | Additional contextual relevance |
| **Total** | **20.0** | |

---

*Report generated: 2026-01-28*
