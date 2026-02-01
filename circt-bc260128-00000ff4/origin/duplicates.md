# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 2 |
| Top Similarity Score | 90 |
| **Recommendation** | **review_existing** |
| Confidence | high |

## Search Parameters

| Parameter | Value |
|-----------|-------|
| Dialect | Arc |
| Failing Pass | LowerStatePass (arc-lower-state) |
| Crash Type | assertion |
| Error Message | state type must have a known bit width; got '!llhd.ref<i1>' |

## Keywords Searched

```
arcilator, StateType, llhd.ref, inout, bit width, LowerState, computeLLVMBitWidth
```

## Search Queries

1. `arcilator inout`
2. `StateType llhd.ref`
3. `inout port Arc`
4. `Arc StateType`
5. `llhd.ref`
6. `LowerState`

## Top Similar Issues


## Recommendation Analysis

**Recommendation**: `review_existing`

### Top Match: Issue #8825

**Title**: [LLHD] Switch from hw.inout to a custom signal reference type

**Similarity Score**: 90/100

**Analysis**:
- ✅ **Excellent match on llhd.ref**: Score includes 30 points for "llhd" keyword match
- ✅ **Strong match on inout ports**: Score includes 20 points for "inout" matches
- ✅ **Related to signal types**: Discusses reference types which is central to this bug
- ⚠️ **Different focus**: Issue #8825 discusses signal reference types at LLHD level, while current bug manifests in Arc StateType

### Second Match: Issue #4916

**Title**: [Arc] LowerState: nested arc.state get pulled in wrong clock tree

**Similarity Score**: 10/100

**Analysis**:
- Matches on "Arc" dialect
- Matches on "LowerState" transformation
- Different root cause (clock tree vs StateType verification)

## Recommendation Details

**Action**: `review_existing`

The high similarity score (90) on Issue #8825 suggests this is a **highly related** issue. However, the exact manifestation differs:

- **Issue #8825**: Focuses on LLHD's reference type design
- **Current Bug**: Arc's StateType cannot handle llhd.ref types in verification

**Next Steps**:
1. ✅ Review Issue #8825 to understand llhd.ref type design
2. ✅ Check if StateType's computeLLVMBitWidth needs llhd::RefType support
3. ✅ Consider if this is:
   - A duplicate (if already being addressed in #8825)
   - A dependent issue (if #8825 fixes the underlying type issue)
   - A separate bug (if Arc-specific StateType handling is needed independently)

## Scoring Weights

| Factor | Weight | Calculation |
|--------|--------|-------------|
| StateType keyword | 30-50 | Per occurrence in title/body |
| llhd.ref keyword | 30-55 | Per occurrence in title/body |
| inout keyword | 20-35 | Per occurrence in title/body |
| bit width keyword | 15-25 | Per occurrence in title/body |
| Arc dialect | 10 | If labeled with Arc |
| LowerState/verify | 10-15 | Per occurrence |
| arcilator tool | 10 | If mentioned |

## Conclusion

⚠️ **Review Issue #8825** before proceeding. While it's not a complete duplicate, it's highly related and may:
- Provide solutions for this bug
- Need to be extended to handle Arc StateType
- Be a prerequisite for fixing this issue


### Issue #8825 - Switch from hw.inout to a custom signal reference type

**URL**: https://github.com/llvm/circt/issues/8825

**Similarity Score**: 90/100

**State**: Open

---

### Issue #4916 - [Arc] LowerState: nested arc.state get pulled in wrong clock tree

**URL**: https://github.com/llvm/circt/issues/4916

**Similarity Score**: 10/100

**State**: Open

---
