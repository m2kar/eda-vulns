# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 24 |
| Top Similarity Score | 9.0 |
| **Recommendation** | **LIKELY_NEW** |
| Confidence | MEDIUM |

## Search Parameters

- **Dialect**: Moore/HW/Arc
- **Failing Pass**: arcilator (suspected)
- **Crash Type**: timeout
- **Timeout**: 60s
- **Keywords**: arcilator, struct, timeout, packed, type conversion, bitcast, moore, hw, non-terminating, port, module instantiation, packed struct, type coercion

## Top Similar Issues

### [#6373](https://github.com/llvm/circt/issues/6373) - Score: 9.0

**Title**: [Arc] Support hw.wires of aggregate types

**State**: OPEN

**Labels**: Arc

**Created**: 2023-11-02T15:55:35Z

---

### [#2329](https://github.com/llvm/circt/issues/2329) - Score: 7.5

**Title**: [LowerToHW] Use type decl for Bundle type lowering

**State**: OPEN

**Labels**: enhancement, ExportVerilog

**Created**: 2021-12-12T22:45:06Z

---

### [#7535](https://github.com/llvm/circt/issues/7535) - Score: 7.0

**Title**: [MooreToCore] VariableOp lowered failed

**State**: OPEN

**Labels**: 

**Created**: 2024-08-20T01:51:29Z

---

### [#8283](https://github.com/llvm/circt/issues/8283) - Score: 7.0

**Title**: [ImportVerilog] Cannot compile forward decleared string type

**State**: OPEN

**Labels**: 

**Created**: 2025-02-28T08:51:12Z

---

### [#3853](https://github.com/llvm/circt/issues/3853) - Score: 6.5

**Title**: [ExportVerilog] Try to make bind change the generated RTL as little as possible

**State**: OPEN

**Labels**: 

**Created**: 2022-09-09T16:20:20Z

---

## Recommendation

**Action**: `likely_new`

**Confidence**: MEDIUM

**Reason**: Related issues found but differences suggest new bug


### 🟡 Proceed with Caution

Related issues exist but this appears to be a different bug.

**Recommended:**
- Proceed to generate the bug report
- Reference related issues in the report
- Highlight what makes this bug different
