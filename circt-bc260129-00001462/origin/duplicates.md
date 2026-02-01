# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Total Issues Found | 49 |
| Top Similarity Score | 17 / 40 |
| **Recommendation** | **LIKELY_NEW** |
| Confidence | MEDIUM |

## Search Context

**Error**: concurrent assertion statements with action blocks are not supported yet

**Keywords Searched**:
- concurrent assertion
- action block
- assert property
- else $error
- ImportVerilog
- not supported yet
- ConcurrentAssertionStatement

**Component**: ImportVerilog

**Dialect**: Moore

**Crash Type**: feature_limitation

## Top Issue

### Issue #9234: [ImportVerilog] Functionality for real number format specifiers not defined

- **State**: OPEN
- **Similarity Score**: 17 / 40
- **Score Breakdown**: keyword_match(2), component_match(10), label_match(5)
- **Created**: 2025-11-17T01:43:55Z
- **URL**: https://github.com/llvm/circt/issues/9234

**Relevance**: This issue has the highest similarity score. Reviewing it is recommended before creating a new report.

## Top 5 Other Related Issues

1. **[#9206](https://github.com/llvm/circt/issues/9206)**: [ImportVerilog] moore.conversion generated instead of moore.int_to_string
   - Score: 17/40
   - State: OPEN

2. **[#8173](https://github.com/llvm/circt/issues/8173)**: [ImportVerilog] Crash on ordering-methods-reverse test
   - Score: 17/40
   - State: OPEN

3. **[#8021](https://github.com/llvm/circt/issues/8021)**: [ImportVerilog] Support handling the slang::ast::StatementBlockSymbol.
   - Score: 17/40
   - State: OPEN

4. **[#7801](https://github.com/llvm/circt/issues/7801)**: [ImportVerilog] How to implement SVA in Moore?
   - Score: 17/40
   - State: OPEN

5. **[#6776](https://github.com/llvm/circt/issues/6776)**: [ImportVerilog] Make AST traveral non-recursive
   - Score: 17/40
   - State: OPEN

## Assessment

**Recommendation**: `likely_new`

**Action Required**:

Related issues exist but address different aspects or components. You can proceed with creating a new issue, but:
1. Reference the related issues in your report
2. Clearly explain what makes this issue different
3. Provide your specific test case that demonstrates the limitation

## Scoring Methodology

| Factor | Max Points | Description |
|--------|-----------|-------------|
| Keyword Match | 10 | Presence of keywords in title/body |
| Error Message Match | 10 | Presence of error-specific terms |
| Component Match | 10 | Matching ImportVerilog component |
| Construct Match | 10 | Matching concurrent assertion constructs |
| **Total Scale** | **40** | Sum of all factors |

---

Generated: 2026-02-01T07:35:18.474327
