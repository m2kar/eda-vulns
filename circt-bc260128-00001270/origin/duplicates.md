# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 3 |
| Top Similarity Score | 2.5 |
| **Recommendation** | **new_issue** |

## Search Parameters

- **Dialect**: Arc
- **Failing Pass**: LowerState
- **Crash Type**: Assertion
- **Keywords**: arcilator, StateType, LLHD reference, inout port, tri-state, verifyInvariants, LowerState pass, !llhd.ref<i1>, bit width

## Related Issues Found

### [#8825](https://github.com/llvm/circt/issues/8825) (Score: 2.0)

**Title**: [LLHD] Switch from hw.inout to a custom signal reference type

**State**: OPEN

**Labels**: LLHD

**Match Reason**: Dialect label match (LLHD)

**Analysis**: This issue discusses moving from `hw.inout` to custom signal reference types. It's related to the general problem area (inout ports and reference types) but does not describe the specific assertion failure or the tri-state buffer issue.

---

### [#9467](https://github.com/llvm/circt/issues/9467) (Score: 2.5)

**Title**: [circt-verilog][arcilator] `arcilator` fails to lower `llhd.constant_time` generated from simple SV delay (`#1`)

**State**: OPEN

**Labels**: LLHD, Arc

**Match Reason**: Dialect label match (LLHD + Arc)

**Analysis**: This is also an arcilator crash related to LLHD types, but it involves `llhd.constant_time` from delays rather than `llhd.ref` from inout ports. Different crash pattern and assertion message.

---

### [#4916](https://github.com/llvm/circt/issues/4916) (Score: 2.0)

**Title**: [Arc] LowerState: nested arc.state get pulled in wrong clock tree

**State**: OPEN

**Labels**: Arc

**Match Reason**: Dialect label and pass match (Arc + LowerState)

**Analysis**: This issue involves the LowerState pass but has a different problem (clock tree handling) and does not involve LLHD reference types or StateType verification failures.

## Recommendation

**Action**: `new_issue`

✅ **Clear to Proceed - Historical Bug Report**

Based on duplicate analysis:

1. **No exact match found**: None of the existing issues describe the same assertion message ("state type must have a known bit width; got '!llhd.ref<i1>'")

2. **Different crash patterns**: While there are other Arc/LowerState/LLHD issues, they involve different:
   - Assertion messages
   - Trigger conditions (delays vs inout+tri-state)
   - Type handling (constant_time vs reference types)

3. **Historical context**: This was a bug in CIRCT 1.139.0 that has been fixed in the current toolchain. The report should document this for historical reference and regression testing.

4. **Recommended action**: Generate issue report with:
   - Tag indicating historical/fixed status
   - Full root cause analysis
   - Test case for regression testing
   - Reference to related issues (#8825, #9467, #4916)

## Scoring Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
| Failing pass match | 2.0 | If failing pass appears in issue |

**Note**: Top score of 2.5 is relatively low, indicating no strong duplicate candidates.
