# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Searched | 14 |
| Top Similarity Score | **5.5** |
| **Recommendation** | **likely_new** |

## Search Parameters

- **Dialect**: Arc
- **Failing Pass**: LowerStatePass
- **Crash Type**: assertion
- **Crash Signature**: `state type must have a known bit width; got '!llhd.ref<i6>'`
- **Keywords**: arcilator, inout, LowerState, StateType, llhd.ref, bit width, assertion, ModuleLowering, computeLLVMBitWidth

## Search Queries Executed

1. `arcilator inout` - No results
2. `StateType llhd.ref` - No results
3. `LowerState assertion` - No results
4. `arcilator bit width` - 1 result (unrelated)
5. `arcilator crash` - 1 result
6. `computeLLVMBitWidth` - No results
7. `label:Arc` - 13 results
8. `inout port arcilator` - No results
9. `llhd.ref` - 1 result

## Top Similar Issues

### [#4916](https://github.com/llvm/circt/issues/4916) (Score: 5.5)

**Title**: [Arc] LowerState: nested arc.state get pulled in wrong clock tree

**State**: OPEN

**Matched Keywords**: title:LowerState, failing_pass, label:Arc

**Analysis**: This issue is about `LowerState` pass but concerns clock tree handling, not `llhd.ref` type handling. **Different root cause**.

---

### [#9467](https://github.com/llvm/circt/issues/9467) (Score: 4.5)

**Title**: [circt-verilog][arcilator] `arcilator` fails to lower `llhd.constant_time` generated from simple SV delay (`#1`)

**State**: OPEN

**Matched Keywords**: body:arcilator, label:Arc, title:arcilator

**Analysis**: This is about `llhd.constant_time` not being handled in ConvertToArcs pass. **Different issue** (time type vs ref type).

---

### [#9260](https://github.com/llvm/circt/issues/9260) (Score: 4.5)

**Title**: Arcilator crashes in Upload Release Artifacts CI

**State**: OPEN

**Matched Keywords**: body:arcilator, label:Arc, title:arcilator

**Analysis**: CI-related crash, unrelated to our bug.

---

### [#8825](https://github.com/llvm/circt/issues/8825) (Score: 4.0)

**Title**: [LLHD] Switch from hw.inout to a custom signal reference type

**State**: OPEN

**Matched Keywords**: body:llhd.ref, body:inout, title:inout

**Analysis**: This is a **feature request** about switching to `llhd.ref<T>` type system. Related to how `inout` ports are represented, but not a crash report. May be **root cause context** for our bug.

---

### [#6810](https://github.com/llvm/circt/issues/6810) (Score: 3.5)

**Title**: [Arc] Add basic assertion support

**State**: OPEN

**Matched Keywords**: title:assertion, label:Arc

**Analysis**: About adding `verif.assert` support, unrelated.

---

## Recommendation

**Action**: `likely_new`

📋 **Proceed with Caution**

Related issues exist involving the Arc dialect and LowerState pass, but this appears to be a **distinct bug**:

1. **No existing issue** reports the specific assertion: `state type must have a known bit width; got '!llhd.ref<i6>'`
2. **No existing issue** describes crashes when arcilator encounters `inout` ports
3. **Issue #8825** provides context - the `llhd.ref` type is a newer addition, and arcilator may not handle it properly yet

### Key Differences from Similar Issues

| Our Bug | #4916 | #9467 | #8825 |
|---------|-------|-------|-------|
| Crash on `llhd.ref` type | Clock tree issue | `llhd.constant_time` issue | Feature request |
| `LowerState.cpp:219` | Different location | `ConvertToArcs` pass | N/A |
| `inout` port trigger | `arc.state` trigger | `#1` delay trigger | N/A |

### Next Steps

1. ✅ Proceed to generate the bug report
2. 📝 Reference #8825 as related context (llhd.ref type system)
3. 🏷️ Add labels: `Arc`, `bug`

## Scoring Weights Used

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If exact assertion appears in body |
| Dialect label match | 1.5 | If Arc label is present |
| Failing pass match | 2.0 | If LowerState appears in issue |
