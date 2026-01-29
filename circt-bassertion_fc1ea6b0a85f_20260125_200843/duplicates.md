# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 6 |
| Top Similarity Score | 5.5 |
| **Recommendation** | **likely_new** |

## Search Parameters

- **Dialect**: Arc
- **Failing Pass**: LowerState
- **Crash Type**: assertion
- **Assertion Message**: `state type must have a known bit width; got '!llhd.ref<i1>'`
- **Keywords**: arcilator, inout, StateType, llhd.ref, LowerState, known bit width, bidirectional port

## Search Queries

1. `arcilator inout` - 0 results
2. `StateType "known bit width"` - 0 results
3. `LowerState inout` - 0 results
4. `llhd.ref` - 1 result
5. `label:Arc` - 13 results

## Top Similar Issues

### [#4916](https://github.com/llvm/circt/issues/4916) (Score: 5.5)

**Title**: [Arc] LowerState: nested arc.state get pulled in wrong clock tree

**State**: OPEN

**Labels**: Arc

**Matched Keywords**: title:LowerState, label:Arc, pass:LowerState

**Relevance**: This issue mentions LowerState pass but is about clock tree issues, NOT about inout ports or StateType bit width assertions. **Different bug.**

---

### [#8825](https://github.com/llvm/circt/issues/8825) (Score: 5.0)

**Title**: [LLHD] Switch from hw.inout to a custom signal reference type

**State**: OPEN

**Labels**: LLHD

**Matched Keywords**: title:inout, body:inout, body:llhd.ref, assertion:llhd.ref

**Relevance**: This is a **feature request** to introduce `!llhd.ref<T>` type. Related to our crash (which involves `!llhd.ref<i1>`), but this issue is about LLHD dialect design, not Arc/arcilator crashes. **Related but different.**

---

### [#9467](https://github.com/llvm/circt/issues/9467) (Score: 4.5)

**Title**: [circt-verilog][arcilator] `arcilator` fails to lower `llhd.constant_time` generated from simple SV delay (`#1`)

**State**: OPEN

**Labels**: LLHD, Arc

**Matched Keywords**: title:arcilator, body:arcilator, label:Arc

**Relevance**: Also an arcilator lowering failure, but caused by `llhd.constant_time`, not inout ports. **Different bug.**

---

### [#9260](https://github.com/llvm/circt/issues/9260) (Score: 4.5)

**Title**: Arcilator crashes in Upload Release Artifacts CI

**State**: OPEN

**Labels**: bug, Arc

**Matched Keywords**: title:arcilator, body:arcilator, label:Arc

**Relevance**: CI-related arcilator crash, unrelated to inout port handling. **Different bug.**

---

### [#8484](https://github.com/llvm/circt/issues/8484) (Score: 4.5)

**Title**: [arcilator] Add functionality to the arcilator runtime library

**State**: OPEN

**Labels**: Arc

**Matched Keywords**: title:arcilator, body:arcilator, label:Arc

**Relevance**: Feature request about runtime library. **Unrelated.**

---

### [#9337](https://github.com/llvm/circt/issues/9337) (Score: 3.5)

**Title**: [arcilator] The position of the passthrough calling

**State**: OPEN

**Labels**: Arc

**Matched Keywords**: title:arcilator, label:Arc

**Relevance**: About passthrough ordering. **Unrelated.**

---

## Analysis

No existing issue directly addresses the crash we found:
- **Crash**: `LowerState.cpp:219` assertion failure when processing inout ports
- **Cause**: `!llhd.ref<i1>` type cannot create `StateType` (unknown bit width)
- **Trigger**: SystemVerilog `inout` port processed by arcilator

The most related issue is **#8825** which discusses creating `!llhd.ref<T>` type, but that's about LLHD dialect design, not Arc dialect's handling of such types.

## Recommendation

**Action**: `likely_new`

📋 **Proceed with Caution**

Related issues exist (especially #8825 about llhd.ref type) but this appears to be a **new bug** specific to:
- Arc dialect's LowerState pass
- Missing early validation for inout ports
- StateType not handling llhd.ref types

**Recommended**:
- Proceed to generate the bug report
- Reference #8825 in the report (related to llhd.ref type)
- Highlight that this is an Arc dialect issue, not LLHD

## Scoring Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
| Failing pass match | 2.0 | If failing pass appears in issue |
