# Duplicate Check Report

## Summary

**Recommendation**: `new_issue` - No existing issues found that match this specific bug.

**Top Similarity Score**: 6.5 (below threshold of 8.0)

**Search Date**: 2026-01-28T11:16:30+00:00

## Bug Signature

- **Error Message**: `state type must have a known bit width; got '!llhd.ref<i1>'`
- **Dialect**: Arc
- **Failing Pass**: LowerStatePass
- **Crash Location**: LowerState.cpp:219
- **Keywords**: arcilator, LowerState, StateType, inout, RefType, llhd.ref, bit width, LLHD

## Search Strategy

Executed 10 search queries against llvm/circt repository:

1. `arcilator inout` - 0 results
2. `LowerState` - 2 results
3. `llhd.ref` - 2 results
4. `state type must have a known bit width` - 0 results
5. `arcilator RefType` - 1 result
6. `StateType` - 2 results
7. `arcilator` - 15 results
8. `bit width` - 10 results
9. `inout port` - 10 results
10. `Arc assertion` - 4 results

**Total unique issues analyzed**: 28

## Top 5 Most Similar Issues

### 1. [#8825](https://github.com/llvm/circt/issues/8825) - [LLHD] Switch from hw.inout to a custom signal reference type
- **State**: Open
- **Similarity Score**: 6.5
- **Matched Keywords**: llhd.ref, RefType, LLHD, inout
- **Analysis**: This issue proposes creating an `llhd.ref<T>` type to represent signal references. It discusses why `hw.inout` is limited and why `llhd::RefType` is needed. **Related** to our bug as it discusses the same `llhd.ref` type that causes our assertion failure, but this is a design discussion/feature request, not a bug report for the specific crash we're seeing.

### 2. [#4916](https://github.com/llvm/circt/issues/4916) - [Arc] LowerState: nested arc.state get pulled in wrong clock tree
- **State**: Open
- **Similarity Score**: 5.0
- **Matched Keywords**: Arc, LowerState, arc.state
- **Analysis**: Bug in LowerState pass where nested `arc.state` operations are pulled into wrong clock tree. Same pass (LowerState) but **different issue** - this is about nested states vs our RefType handling problem.

### 3. [#5053](https://github.com/llvm/circt/issues/5053) - [Arc] LowerState: combinatorial cycle reported in cases where there is none
- **State**: Closed
- **Similarity Score**: 4.5
- **Matched Keywords**: Arc, LowerState
- **Analysis**: LowerState incorrectly reports combinatorial cycles. **Different bug type** but same pass involved. Already closed.

### 4. [#9467](https://github.com/llvm/circt/issues/9467) - [circt-verilog][arcilator] arcilator fails to lower llhd.constant_time
- **State**: Open
- **Similarity Score**: 4.0
- **Matched Keywords**: arcilator, LLHD, failed to legalize
- **Analysis**: Arcilator cannot lower `llhd.constant_time` from SV delays. **Similar pattern** (arcilator failing on LLHD ops) but different specific operation. Our issue is about `llhd.ref` in LowerState, not `llhd.constant_time` in ConvertToArcs.

### 5. [#8012](https://github.com/llvm/circt/issues/8012) - [Moore][Arc][LLHD] Moore to LLVM lowering issues
- **State**: Open
- **Similarity Score**: 4.0
- **Matched Keywords**: arcilator, LLHD, Arc
- **Analysis**: Arcilator fails with `llhd.process` not supported by ConvertToArcs. **Similar issue category** (arcilator + LLHD unsupported ops) but different specific problem and different pass.

## Detailed Comparison

| Issue | Crash Location | Error Type | Root Cause Match |
|-------|---------------|------------|------------------|
| Our Bug | LowerState.cpp:219 | Assertion (verifyInvariants) | llhd::RefType not supported by StateType |
| #8825 | N/A (design discussion) | N/A | ✓ Discusses llhd.ref design |
| #4916 | LowerState (clock tree) | Logic error | ✗ Different cause |
| #5053 | LowerState (cycle detection) | False positive | ✗ Different cause |
| #9467 | ConvertToArcs | Legalization failure | ✗ Different op, different pass |
| #8012 | ConvertToArcs | Unsupported region | ✗ Different op, different pass |

## Conclusion

**This appears to be a NEW BUG** that has not been reported before.

### Key Differentiators:
1. **Unique Error Message**: "state type must have a known bit width; got '!llhd.ref<i1>'" - No existing issue mentions this exact error.
2. **Specific Crash Location**: LowerState.cpp:219 in StateType::get() - Not reported in any existing issue.
3. **Root Cause**: `llhd::RefType` (from inout ports) is not supported by `arc::StateType::computeLLVMBitWidth()` - This specific interaction is not documented in any existing issue.

### Related Issue to Reference:
- **#8825** should be referenced in the new bug report, as it discusses the `llhd.ref` type design and may provide context for the fix direction.

## Recommendation

**Proceed with creating a new GitHub Issue.** The bug is distinct from all existing issues. Reference #8825 for context on `llhd.ref` type design.
