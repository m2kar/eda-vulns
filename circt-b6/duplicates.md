# Duplicate Check Report: circt-b6

**Check Date**: 2026-02-01T05:15:00Z
**Recommendation**: **NEW ISSUE**
**Confidence**: High

## Summary

No duplicate issues found in the CIRCT repository. This appears to be a **new, unreported bug**.

## Search Strategy

Searched llvm/circt GitHub Issues with the following queries:
1. "Arc StateType" - 0 relevant results
2. "LowerState assertion" - 0 relevant results
3. "LLHD inout" - 10 results (analyzed below)
4. "arcilator crash" - 2 results (analyzed below)
5. "inout always_ff" - 10 results (none relevant)
6. "llhd.ref" - 2 results (analyzed below)
7. "StateType bit width" - 1 result (not relevant)

## Related Issues (Not Duplicates)

### Issue #8825: [LLHD] Switch from hw.inout to a custom signal reference type
- **State**: OPEN
- **Similarity Score**: 7.5/10
- **Type**: Feature request
- **Relationship**: Discusses introducing `!llhd.ref<T>` type (which is what causes our crash), but this is a design discussion about the type system, not a bug report about crashes when using inout ports.
- **Why Not Duplicate**: This is a feature request to introduce the reference type system. Our bug is about Arc dialect crashing when it encounters these reference types in sequential logic.

### Issue #9395: [circt-verilog][arcilator] Arcilator assertion failure (CLOSED)
- **State**: CLOSED (2026-01-19)
- **Similarity Score**: 3.0/10
- **Type**: Bug report
- **Relationship**: Also an arcilator assertion failure, but different root cause
- **Why Not Duplicate**: That issue was about `always @*` combinational blocks with assertions. Our issue is about inout ports in `always_ff` sequential blocks causing StateType verification failure.

### Issue #9260: Arcilator crashes in Upload Release Artifacts CI
- **State**: OPEN
- **Similarity Score**: 2.0/10
- **Type**: CI infrastructure issue
- **Why Not Duplicate**: CI-specific crash, not related to inout ports or StateType.

## Key Differentiators

Our bug is unique because it involves:
1. **Specific construct**: Inout ports used in `always_ff` sequential logic
2. **Specific crash point**: `LowerState.cpp:219` in Arc dialect
3. **Specific error**: "state type must have a known bit width; got '!llhd.ref<i1>'"
4. **Type system issue**: Arc StateType cannot handle LLHD reference types

No existing issue reports this specific combination.

## Search Coverage

- **Total issues searched**: ~50 across 7 queries
- **Relevant issues examined**: 3
- **Duplicates found**: 0
- **Related issues**: 1 (feature request #8825)

## Recommendation

**Proceed with new issue submission**. This is a legitimate, unreported bug that should be filed with the CIRCT project.

**Suggested Title**: `[Arc] Assertion failure when lowering inout ports in sequential logic`

**Suggested Labels**: `bug`, `Arc`, `LLHD`, `type-system`

**Related Issue**: Mention issue #8825 in the bug report as context (the feature request discusses the LLHD reference type system that causes this crash).
