# Duplicate Check Report

## Search Summary

| Query | Results |
|-------|---------|
| "string port crash" | 5 issues found |
| "MooreToCore string" | 5 issues found |
| "DynamicStringType port" | 0 issues found |

## Potential Duplicates

### 🔴 HIGH SIMILARITY: Issue #8332 (OPEN)
**Title**: [MooreToCore] Support for StringType from moore to llvm dialect

**URL**: https://github.com/llvm/circt/issues/8332

**Similarity Score**: 85%

**Analysis**:
- This issue directly addresses the lack of StringType support in MooreToCore conversion
- The issue author is working on adding string type support to the sim dialect and lowering to LLVM
- Our crash is a **symptom** of the same underlying problem: StringType is not properly handled during conversion

**Relationship**: **SAME ROOT CAUSE** - The fix for #8332 would resolve our crash

### 🟡 RELATED: Issue #8283 (OPEN)
**Title**: [ImportVerilog] Cannot compile forward declared string type

**URL**: https://github.com/llvm/circt/issues/8283

**Similarity Score**: 75%

**Analysis**:
- Discusses similar string type handling failures in Moore dialect
- Different manifestation (variable declaration vs port), but related infrastructure issue

**Relationship**: Related but different specific issue

## Recommendation

### ⚠️ DO NOT SUBMIT AS NEW ISSUE

**Reason**: Our crash is a manifestation of the known issue #8332. The lack of proper StringType support in MooreToCore causes:
1. Issue #8283: String variables fail to compile
2. Issue #8332: Overall StringType support missing
3. **Our case**: String ports crash during module conversion

### Suggested Action

Instead of creating a new issue, consider adding a comment to **Issue #8332** with:
- Our specific crash scenario (string type output port)
- The crash location (`getModulePortInfo()` in MooreToCore.cpp)
- Our minimized test case

This additional information could help prioritize and guide the fix for the underlying StringType support issue.

## Summary

| Finding | Value |
|---------|-------|
| Duplicate Found | ✅ Yes |
| Primary Duplicate | #8332 |
| Should Submit New Issue | ❌ No |
| Alternative Action | Comment on #8332 |
