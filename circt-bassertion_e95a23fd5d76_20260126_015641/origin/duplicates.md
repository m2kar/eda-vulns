# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 8 |
| Top Similarity Score | 7.5 |
| **Recommendation** | **likely_new** |

## Search Parameters

- **Dialect**: Arc
- **Failing Pass**: LowerStatePass
- **Crash Type**: assertion
- **Keywords**: arcilator, inout, StateType, llhd.ref, LowerState, computeLLVMBitWidth, bit width, RefType
- **Assertion Message**: `state type must have a known bit width; got '!llhd.ref<i1>'`

## Top Similar Issues

### [#8825](https://github.com/llvm/circt/issues/8825) (Score: 7.5)

**Title**: [LLHD] Switch from hw.inout to a custom signal reference type

**State**: OPEN

**Labels**: LLHD

**Match Reasons**:
- Directly discusses llhd.ref type
- Mentions signal reference handling
- Related to type system changes affecting arcilator

**Relevance**: This issue proposed the `!llhd.ref<T>` type that is now causing the crash. The issue discusses the need for a custom reference type but doesn't address arcilator compatibility.

---

### [#8286](https://github.com/llvm/circt/issues/8286) (Score: 5.0)

**Title**: [circt-verilog][llhd][arcilator] Verilog-to-LLVM lowering issues

**State**: OPEN

**Labels**: None

**Match Reasons**:
- Discusses arcilator lowering failures
- Related to circt-verilog pipeline
- General lowering issues

**Relevance**: General discussion of arcilator lowering issues, but doesn't specifically mention llhd.ref or StateType.

---

### [#8012](https://github.com/llvm/circt/issues/8012) (Score: 4.5)

**Title**: [Moore][Arc][LLHD] Moore to LLVM lowering issues

**State**: OPEN

**Labels**: None

**Match Reasons**:
- Discusses arcilator failures
- Related to Moore/LLHD lowering
- Pipeline issues

**Relevance**: Discusses arcilator failures with different error messages (llhd.process, seq.clock_inv).

---

### [#4916](https://github.com/llvm/circt/issues/4916) (Score: 4.0)

**Title**: [Arc] LowerState: nested arc.state get pulled in wrong clock tree

**State**: OPEN

**Labels**: Arc

**Match Reasons**:
- LowerState pass related
- Arc dialect issue

**Relevance**: Related to LowerState pass but different issue (clock tree handling).

---

### [#5566](https://github.com/llvm/circt/issues/5566) (Score: 3.5)

**Title**: [SV] Crash in `P/BPAssignOp` verifiers for `hw.inout` ports

**State**: OPEN

**Labels**: bug, good first issue, Verilog/SystemVerilog

**Match Reasons**:
- hw.inout port crash
- Similar assertion failure pattern

**Relevance**: Different crash location (SV dialect verifiers) but similar pattern of inout port handling issues.

---

## Related Pull Requests

### [PR #9081](https://github.com/llvm/circt/pull/9081) - **HIGHLY RELEVANT**

**Title**: [LLHD] Add new RefType to replace hw::InOutType

**State**: MERGED (2025-10-12)

**Description**: This PR introduced the `!llhd.ref<T>` type that is now causing the crash. The PR replaced `hw::InOutType` with `llhd::RefType` across the LLHD dialect and ImportVerilog pipeline.

**Key Quote from PR**:
> Add a `RefType` to the LLHD dialect and replace all uses of HW's `InOutType` across the LLHD dialect and the ImportVerilog pipeline.

**Analysis**: This PR appears to have introduced the `llhd.ref` type but may not have updated arcilator's `LowerState` pass to handle it. The `computeLLVMBitWidth()` function in `ArcTypes.cpp` only handles `ClockType`, `IntegerType`, `ArrayType`, and `StructType` - not `llhd::RefType`.

---

## Recommendation

**Action**: `likely_new`

**Confidence**: Medium

### Reasoning

1. **No exact duplicate found**: No existing issue specifically reports arcilator's `LowerState` pass failing on `!llhd.ref<T>` types with the error "state type must have a known bit width".

2. **Related but different**: Issue #8825 discusses the design of `llhd.ref` type, but doesn't report this specific crash.

3. **Root cause identified**: PR #9081 introduced `llhd::RefType` but arcilator's `computeLLVMBitWidth()` function was not updated to handle it.

4. **Gap in implementation**: This appears to be a missing case in the arcilator lowering pipeline after the RefType introduction.

### Suggested Action

**Proceed to create a new issue** with the following notes:
- Reference Issue #8825 (llhd.ref type design)
- Reference PR #9081 (RefType implementation)
- Highlight that `computeLLVMBitWidth()` in `ArcTypes.cpp` needs to handle `llhd::RefType`
- This is likely a follow-up fix needed after the RefType migration

---

## Scoring Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
| Failing pass match | 2.0 | If failing pass appears in issue |
