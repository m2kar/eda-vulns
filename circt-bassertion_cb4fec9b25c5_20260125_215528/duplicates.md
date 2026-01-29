# Duplicate Check Report

## Search Summary

Searched GitHub issues in `llvm/circt` repository with queries:
- "string port"
- "MooreToCore string"  
- "dyn_cast non-existent"

## Duplicate Found ✅

### Primary Match: Issue #8332

| Field | Value |
|-------|-------|
| Title | [MooreToCore] Support for StringType from moore to llvm dialect |
| State | OPEN |
| URL | https://github.com/llvm/circt/issues/8332 |
| Created | 2025-03-20 |
| Similarity | **Exact Match** |

**Match Reason**: This issue directly addresses the lack of StringType support in the MooreToCore conversion, which is exactly what causes our crash.

### Related Issue: #8283

| Field | Value |
|-------|-------|
| Title | [ImportVerilog] Cannot compile forward declared string type |
| State | OPEN |
| URL | https://github.com/llvm/circt/issues/8283 |
| Similarity | Related |

## Recommendation

⛔ **DO NOT SUBMIT** - This crash is a known limitation already tracked in issue #8332.

The community is aware of the missing StringType support and has an open issue for it. Submitting a new issue would be a duplicate.
