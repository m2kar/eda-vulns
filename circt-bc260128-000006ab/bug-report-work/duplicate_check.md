# Duplicate Check Report

## Search Strategy

Searched llvm/circt repository for issues related to:
1. "inout port arcilator crash"
2. "StateType"
3. "llhd.ref"
4. "arcilator assertion"
5. "known bit width"
6. "LowerState"
7. "inout crash"
8. "tri-state"
9. "Arc crash"
10. "arcilator"

## Related Issues Found

### 1. Issue #8825 - [LLHD] Switch from hw.inout to a custom signal reference type
**Created:** 2025-08-06  
**Status:** OPEN  
**Relevance:** HIGH  
**Description:** Discusses the transition from `!hw.inout` to `!llhd.ref<T>` type system. This is directly related to our error message "got '!llhd.ref<i1>'".

**Connection:** Our crash occurred because StateType received a `!llhd.ref<i1>` type. Issue #8825 proposes the very type system that caused the crash, suggesting this was during the migration period.

### 2. Issue #5566 - [SV] Crash in `P/BPAssignOp` verifiers for `hw.inout` ports
**Created:** 2023-07-12  
**Status:** OPEN  
**Relevance:** MEDIUM  
**Description:** Crash in SystemVerilog dialect verifiers when handling `hw.inout` ports.

**Connection:** Both involve crashes with `hw.inout` ports, but crash location is different (SV dialect vs Arc dialect).

### 3. Issue #4036 - [PrepareForEmission] Crash when inout operations are passed to instance ports
**Created:** 2022-09-30  
**Status:** OPEN  
**Relevance:** MEDIUM  
**Description:** Crash in `StorageUniquerSupport.h` (same file as our crash) when handling inout ports.

**Connection:** Same crash location (`StorageUniquerSupport.h`), but different dialect/phase.

### 4. Issue #9260 - Arcilator crashes in Upload Release Artifacts CI
**Created:** 2025-11-24  
**Status:** OPEN  
**Relevance:** LOW  
**Description:** Arcilator crashes in CI, but no specific connection to our issue.

## Duplicate Analysis

### Similarity Assessment

| Issue | Keywords Match | Root Cause | Code Path | Similarity |
|-------|---------------|------------|-----------|------------|
| #8825 | ✅ (llhd.ref, inout) | Type system | LLHD dialect | HIGH |
| #5566 | ✅ (inout, crash) | Inout handling | SV dialect | MEDIUM |
| #4036 | ✅ (inout, crash, StorageUniquer) | Type creation | Preparation | MEDIUM |
| #9260 | ⚠️ (arcilator, crash) | Unknown | Arcilator | LOW |

### Crash Signature Analysis

**Our Crash:**
```
state type must have a known bit width; got '!llhd.ref<i1>'
StorageUniquerSupport.h:180: StateType::get()
LowerState.cpp:219: ModuleLowering::run()
```

**Key Distinguishing Factors:**
1. Specific error message about "known bit width"
2. Arc dialect LowerState pass
3. LLHD reference type involved
4. StateType validation failure

### Conclusion

**Not a Duplicate** - This is a unique crash with specific characteristics:

1. **Unique Error Message:** "state type must have a known bit width; got '!llhd.ref<i1>'" does not appear in any existing issues
2. **Unique Code Path:** LowerState pass in Arc dialect, not SV or other dialects
3. **Unique Trigger:** Combination of inout ports with tri-state assignments in SystemVerilog
4. **Related but Different:** Issue #8825 discusses the `!llhd.ref<T>` type system but does not document this specific crash

**Relationship to Existing Issues:**
- **Issue #8825** is most relevant as it documents the LLHD reference type transition
- Our crash likely occurred during the migration period addressed by #8825
- The crash may have been a side effect of that type system transition

## Recommendations

1. Submit this as a new issue (not a duplicate)
2. Reference issue #8825 for context on LLHD reference types
3. Note that the bug appears to be fixed in current CIRCT version
4. Suggest investigating if the fix was part of the #8825 work or a separate fix
