# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 2 |
| Top Similarity Score | 4.0 |
| Recommendation | **likely_new** |
| Confidence | medium |

## Search Parameters

- **Dialect**: Arc
- **Failing Pass**: LowerStatePass  
- **Crash Type**: assertion
- **Keywords**: arcilator, inout, llhd.ref, StateType, bit width, LowerState
- **Assertion**: state type must have a known bit width; got '!llhd.ref<i1>'

## Top Similar Issues

### [Issue #8825](https://github.com/llvm/circt/issues/8825)

**Score**: 4.0

**Title**: [LLHD] Switch from hw.inout to a custom signal reference type

**State**: OPEN

**Labels**: LLHD

**Summary**: Discussion about switching from `!hw.inout` to `!llhd.ref<T>` for signal references, particularly for supporting `llhd.time` values in variables. This is closely related to our issue where `!llhd.ref<i1>` is not supported by `StateType`.

---

### [Issue #4916](https://github.com/llvm/circt/issues/4916)

**Score**: 3.5

**Title**: [Arc] LowerState: nested arc.state get pulled in wrong clock tree

**State**: OPEN

**Labels**: Arc

**Summary**: Related to Arc dialect and LowerState pass, but focused on clock tree handling rather than type support.

---

## Recommendation

**Action**: `likely_new`  
**Confidence**: medium  
**Reason**: 找到相关 Issue 但可能是新 Bug

### Analysis Details

The search found 2 related issues:

1. **Issue #8825** (Score: 4.0) - Related to `llhd.ref` type but focused on switching from `hw.inout` rather than StateType support
2. **Issue #4916** (Score: 3.5) - Arc dialect and LowerState pass but different problem domain

Neither issue directly addresses the core problem: **StateType's `computeLLVMBitWidth()` function cannot handle `!llhd.ref<T>` types, causing assertion failure in LowerStatePass.**

### Scoring Breakdown

| Factor | Weight | Notes |
|--------|--------|-------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | Exact assertion text matching |
| Dialect label match | 1.5 | Arc label match |
| Failing pass match | 2.0 | LowerState/LowerStatePass mention |

### Score Interpretation

- **Score >= 8.0**: High similarity - likely duplicate, review existing issue
- **Score 4.0-7.9**: Medium similarity - related but may be different bug
- **Score < 4.0**: Low similarity - likely new issue

---

**Conclusion**: This appears to be a **new bug** worth reporting, though related to ongoing discussions about LLHD reference types. The specific failure case (StateType verification with llhd.ref in LowerStatePass) should be documented.
