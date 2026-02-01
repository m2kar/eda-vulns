# CIRCT GitHub Issues Duplicate Check Report

## Summary
- **Analysis Date**: 2025-01-31
- **Search Queries Executed**: 9
- **Issues Found**: 6 related issues
- **Top Match Score**: 7.5/10 (Issue #9467)
- **Recommendation**: **likely_new** - File as a new issue with reference to related patterns

---

## Search Strategy

Searched GitHub Issues in `llvm/circt` using key patterns extracted from `analysis.json`:

1. **Direct operation names**: sim.fmt.literal, FormatLiteralOp, PrintFormatted
2. **Error patterns**: legalization failure, assertion error, failed to legalize operation
3. **Related context**: immediate assertion, $error format, arcilator conversion, sim dialect
4. **Triggering constructs**: immediate assertions with $error() in always_comb blocks

---

## Analysis Results

### Top Match: Issue #9467 (Score: 7.5/10)
**Title**: [circt-verilog][arcilator] `arcilator` fails to lower `llhd.constant_time`...

**Similarity Rationale**:
- ✅ Same arcilator/LowerArcToLLVM conversion context
- ✅ Same legalization failure pattern (operation marked illegal but not properly converted)
- ✅ Same root cause hypothesis: operation remains unconverted because it's orphaned
- ❌ Different dialect (LLHD vs Sim)
- ❌ Different operation type (llhd.constant_time vs sim.fmt.literal)

**Why Not a Duplicate**:
- The legalization failure pattern is similar but affects different dialects
- The orphaned operation problem appears systematic but manifests differently
- #9467 involves temporal operations in delays, not format strings from assertions
- Different conversion pipeline stage (ConvertToArcs vs LowerArcToLLVM format string handling)

---

### Secondary Matches (Low Relevance)

#### Issue #6810 (Score: 5.0/10)
**Title**: [Arc] Add basic assertion support
- Covers general assertion support in Arc dialect
- NOT specific to immediate assertions with format strings
- Feature request, not a bug report

#### Issue #7692 (Score: 4.0/10)
**Title**: [Sim] Combine integer formatting ops...
- Discusses refactoring sim formatting operations
- Different scope (code organization, not functionality bugs)

#### Issue #8332 (Score: 3.5/10)
**Title**: [MooreToCore] Support for StringType...
- Conversion from Moore to LLVM with strings
- Not related to $error() format strings in assertions

#### Issue #8817 (Score: 3.5/10)
**Title**: [FIRRTL] Support special substitutions in assert intrinsics
- Handles assert intrinsics at FIRRTL level
- Not related to immediate assertions or arcilator

#### Issue #9191 (Score: 2.5/10)
**Title**: MLIR pattern checks failing
- General pattern check issue affecting many tests
- No specific mention of sim.fmt.literal

---

## Scoring Methodology

Each issue was scored based on:
- **Crash type match** (+3 points): legalization_failure
- **Same operation mentioned** (+4 points): sim.fmt.literal
- **Same pattern** (+2 points): immediate assertion / orphaned operation
- **Same file mentioned** (+1 point each): LowerArcToLLVM, SimOps, etc.
- **Error message similarity** (+2 points): "failed to legalize operation"

**Score 8+**: Definite duplicate → file as duplicate
**Score 3-7**: Likely new issue → reference related patterns
**Score <3**: Unrelated

---

## Recommendation: FILE AS NEW ISSUE

### Key Unique Characteristics:

1. **Specific Trigger**: Immediate assertions with `$error()` in `always_comb` blocks
   - Generates `sim.fmt.literal` operations from format strings
   - These operations are orphaned without being consumed by print operations

2. **Root Cause Hypothesis**:
   - Format literals from immediate assertions lack proper consumption or removal
   - LowerArcToLLVM marks `sim::FormatLiteralOp` as legal (expecting DCE)
   - But DCE doesn't remove them because either:
     - They're not marked Pure properly
     - Or they have spurious uses that prevent removal
     - Or the foldFormatString pattern only works through PrintFormattedProcOp paths

3. **Related Pattern** (#9467):
   - Similar architecture: operation marked legal but orphaned
   - Different manifestation: orphaned temporal ops vs orphaned format literals
   - Suggests systematic issue with how "legal but unused" operations are handled

### Suggested Issue Title:
```
[arcilator] Legalization failure: orphaned sim.fmt.literal from immediate assertions with $error()
```

### Suggested Description:
```
## Summary
Immediate assertions with $error() in always_comb blocks cause legalization failure 
when sim.fmt.literal operations remain unconverted after LowerArcToLLVM.

## Related Issue
#9467 - Similar legalization failure pattern with orphaned operations in arcilator

## Key Files
- lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp:1236-1239 (FormatLiteralOp marked legal)
- lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp:805-823 (foldFormatString handling)
```

---

## Search Coverage

| Query | Results | Relevant |
|-------|---------|----------|
| sim.fmt.literal | 0 | N/A |
| legalization failure | 2 | 1 (#9467) |
| assertion error | 50+ | 1 (#6810) |
| FormatLiteralOp | 0 | N/A |
| PrintFormatted | 0 | N/A |
| immediate assertion | 3 | 1 (#6810) |
| $error format | 8 | 1 (#8817) |
| arcilator conversion | 3 | 1 (#9467) |
| sim dialect | 10 | 1 (#7692) |

**Coverage**: No exact matches for sim.fmt.literal or FormatLiteralOp confirms this is likely a NEW issue.

---

## Next Steps

1. ✅ **Verify** the crash is reproducible (already confirmed)
2. ✅ **Confirm** no existing issue matches this exact pattern
3. 📋 **Generate** full GitHub issue report with minimized test case
4. 🚀 **File** issue with #9467 as related reference

