# CIRCT Duplicate Issue Check Report

**Timestamp**: 2026-02-01T08:05:10.203484

## Executive Summary

- **Status**: REVIEW_EXISTING
- **Top Similarity Score**: 15.0
- **Top Related Issue**: #9572
- **Total Issues Found**: 76

## Analyzed Bug Information

- **Test Case ID**: 260129-00001473
- **Dialect**: MooreToCore
- **Crash Type**: assertion
- **Root Cause**: String type module port causes assertion failure in sanitizeInOut() due to invalid dyn_cast on sim::DynamicStringType

## Search Keywords Used

- 1. string port
- 2. InOutType
- 3. MooreToCore
- 4. SVModuleOp
- 5. dyn_cast
- 6. DynamicStringType

## Search Results

### Recommendation: REVIEW_EXISTING


**ACTION**: Review existing GitHub issues, particularly issue #9572

The highest similarity score (15.0) suggests potential duplicates exist.
Please manually verify the top related issues below before filing a new report.


## Top 10 Most Similar Issues

| Rank | Issue | Score | Title |
|------|-------|-------|-------|
| 1 | #9572 | 15.0 | [Moore] Assertion failure when module has string type output... |
| 2 | #9570 | 12.0 | [Moore] Assertion in MooreToCore when module uses packed uni... |
| 3 | #9542 | 4.0 | [Moore] to_builtin_bool should be replaced with to_builtin_i... |
| 4 | #9538 | 4.0 | `hlstool` crashes on handshake input |
| 5 | #9395 | 4.0 | [circt-verilog][arcilator] Arcilator assertion failure |
| 6 | #9376 | 4.0 | Memory leak in HW to SystemC conversion |
| 7 | #9371 | 4.0 | Folding rollback error during HW to LLVM conversion |
| 8 | #9315 | 4.0 | [FIRRTL] ModuleInliner removes NLA referred by circt.tracker |
| 9 | #9258 | 4.0 | FlattenMemRefPass crashes on memref.global |
| 10 | #9563 | 3.0 | [FIRRTL][circt-reduce] extmodule-instance-remover crash w/ p... |

## Matched Keywords in Top Issue #9572

- string port
- InOutType
- MooreToCore
- SVModuleOp
- dyn_cast

## Details

- **Total issues analyzed**: 76
- **Duplication threshold**: 10.0
- **Scores >= 10.0**: 2

## Scoring Methodology

1. Similarity scores based on keyword matching:
   - Direct keyword match: +2 points per keyword
   - Related terms (string, moore, port, type, assertion): +1 point each

2. Score >= 10.0 indicates potential duplicate - manual review recommended

3. GitHub API fallback used if gh CLI search unavailable

---

Generated: 2026-02-01 08:06:55
Test Case: 260129-00001473
