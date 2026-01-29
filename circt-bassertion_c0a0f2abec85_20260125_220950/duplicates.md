# Duplicate Check Report

## Summary

**Recommendation**: `new_issue`

**Confidence**: High

No existing issue closely matches this specific crash scenario.

## Search Details

### Keywords Used
- `inout`, `llhd.ref`, `StateType`, `LowerState`, `arcilator`, `bit width`, `RefType`, `tri-state`, `assertion`

### Assertion Message
```
state type must have a known bit width; got '!llhd.ref<i8>'
```

### Dialect
- Arc (with related: LLHD, Moore, HW)

## Candidate Issues Analysis

| # | Issue | Title | State | Score | Match |
|---|-------|-------|-------|-------|-------|
| 1 | [#8825](https://github.com/llvm/circt/issues/8825) | [LLHD] Switch from hw.inout to a custom signal reference type | open | 4.5/10 | `llhd.ref`, `inout` |
| 2 | [#9395](https://github.com/llvm/circt/issues/9395) | [circt-verilog][arcilator] Arcilator assertion failure | closed | 4.0/10 | `arcilator`, `assertion`, `Arc` |
| 3 | [#9467](https://github.com/llvm/circt/issues/9467) | arcilator fails to lower llhd.constant_time | open | 3.5/10 | `arcilator`, `LLHD`, `Arc` |
| 4 | [#8012](https://github.com/llvm/circt/issues/8012) | [Moore][Arc][LLHD] Moore to LLVM lowering issues | open | 3.5/10 | `arcilator`, `LLHD` |
| 5 | [#8286](https://github.com/llvm/circt/issues/8286) | Verilog-to-LLVM lowering issues | open | 3.0/10 | `arcilator`, `llhd` |

## Detailed Analysis

### Issue #8825 (Score: 4.5/10)
**Title**: [LLHD] Switch from hw.inout to a custom signal reference type

**Relevance**: Discusses `!llhd.ref<T>` type for signal references - the same type that causes our crash.

**Why Not Duplicate**:
- This is a **feature request** for type system changes, not a bug report
- Does not mention assertion failures or crashes
- Does not involve `LowerState` pass or `StateType`
- Our crash is a concrete bug in existing code, not a missing feature

### Issue #9395 (Score: 4.0/10)
**Title**: [circt-verilog][arcilator] Arcilator assertion failure

**Relevance**: Arcilator assertion failure with Arc dialect.

**Why Not Duplicate**:
- Different assertion: `allUsesReplaced` in `DialectConversion.cpp`
- Different crash location: `ConvertToArcs.cpp:537`
- Different trigger: `llhd.combinational` lowering, not inout ports
- Issue is already **closed**

### Issue #8012 (Score: 3.5/10)
**Title**: [Moore][Arc][LLHD] Moore to LLVM lowering issues

**Relevance**: Arcilator lowering issues with LLHD.

**Why Not Duplicate**:
- Different issue: `llhd.process` has regions not supported by ConvertToArcs
- Different error: `seq.clock_inv` failed to legalize
- No mention of inout ports or `!llhd.ref` type
- No StateType bit width assertion

### Issue #8286 (Score: 3.0/10)
**Title**: [circt-verilog][llhd][arcilator] Verilog-to-LLVM lowering issues

**Relevance**: Arcilator/LLHD lowering problems.

**Why Not Duplicate**:
- Different errors: `llhd.constant_time`, `body contains non-pure operation`
- Focus on blocking/nonblocking assignments, not inout ports
- No mention of StateType or bit width assertions

## Conclusion

Our bug is **unique** because:

1. **Specific Crash Point**: Assertion failure in `LowerState.cpp:219` (`ModuleLowering::run()`)
2. **Specific Error**: `state type must have a known bit width; got '!llhd.ref<i8>'`
3. **Specific Trigger**: `inout` ports in SystemVerilog converted to `!llhd.ref` type
4. **Root Cause**: `computeLLVMBitWidth()` in `ArcTypes.cpp` doesn't handle `llhd::RefType`

No existing issue reports this specific failure mode. The closest issue (#8825) is related to `llhd.ref` type design but is a feature discussion, not a bug report.

---

**Top Score**: 4.5/10  
**Top Issue**: #8825  
**Threshold for Duplicate**: 7.0/10  
**Verdict**: Score below threshold → **New Issue Recommended**
