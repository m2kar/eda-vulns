# Duplicate Check Report

## Search Summary
Repository: llvm/circt
Search Date: 2026-01-28

### Queries Performed
| Query | Results | Relevant |
|-------|---------|----------|
| `inout arcilator` | 1 | ❌ |
| `llhd.ref StateType` | 0 | - |
| `arcilator inout` | 1 | ❌ |
| `LowerState arcilator` | 0 | - |
| `state type must have a known bit width` | 1 | ❌ |
| `arcilator crash` | 2 | ❌ |
| `bidirectional` | 7 | ❌ |
| `llhd ref` | 12 | ❌ |

## Potentially Related Issues

### Issue #8825 (OPEN)
**Title**: [LLHD] Switch from hw.inout to a custom signal reference type
**Similarity Score**: 4.0 / 10.0

This issue discusses the design of LLHD reference types (`!llhd.ref<T>`), which is the type causing our crash. However, it's a feature request about type system design, not a bug report about arcilator crashes.

**Different because**: Feature discussion, not crash bug

### Issue #9395 (CLOSED)
**Title**: [circt-verilog][arcilator] Arcilator assertion failure
**Similarity Score**: 3.0 / 10.0

This issue reports an arcilator assertion failure, but the crash occurs in `ConvertToArcs.cpp` when processing `CombinationalOp`, not in `LowerState.cpp` when processing `inout` ports.

**Different because**: Different crash location (CombinationalOp vs LowerState)

### Issue #9052 (CLOSED)
**Title**: [circt-verilog] Import difference of results in arcilator failure
**Similarity Score**: 2.0 / 10.0

This issue discusses LLHD output inconsistency, not crashes.

**Different because**: Output consistency issue, not crash

## Recommendation

**✅ NEW ISSUE** - This appears to be a new, unreported bug.

### Reasoning
1. **Unique Crash Signature**: "state type must have a known bit width; got '!llhd.ref<i1>'" is not found in any existing issues
2. **Unique Crash Location**: `LowerState.cpp:219` in `ModuleLowering::run()` is not reported
3. **Unique Trigger**: `inout` (bidirectional) ports causing the crash is not documented
4. **No Exact Match**: No issues report arcilator failing on `inout` ports specifically

### Keywords for Future Reference
- `inout` port
- `llhd.ref` type
- `arc::StateType`
- `LowerState` pass
- `arcilator` crash
- bidirectional
