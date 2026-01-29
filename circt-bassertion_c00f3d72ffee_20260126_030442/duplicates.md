# Duplicate Check Report

## Search Parameters
- **Keywords**: arcilator, inout, StateType, LowerState, llhd.ref, computeLLVMBitWidth, bidirectional, tristate, RefType
- **Repository**: llvm/circt
- **Assertion**: `state type must have a known bit width; got '!llhd.ref<i8>'`

## Results Summary

Searched llvm/circt repository for existing issues matching the crash signature. Found **6 related issues** but **none are exact duplicates**.

The closest match is issue #8825 which tracks the underlying type system limitation (`llhd.ref` type), but our crash is a distinct bug report about assertion failure when using inout ports with arcilator.

## Top Matches

### #8825 - [LLHD] Switch from hw.inout to a custom signal reference type
- **Similarity Score**: 7.5/15
- **State**: OPEN
- **Labels**: LLHD
- **Matched Keywords**: llhd.ref, inout
- **URL**: https://github.com/llvm/circt/issues/8825
- **Analysis**: This is the tracking issue for implementing a proper `llhd.ref<T>` type to replace `hw.inout`. The issue body explicitly mentions `!llhd.ref<T>` which appears in our error message. However, this is a **feature request** for new type support, not a bug report about the crash we're experiencing. Our crash should be reported separately, referencing this issue as related.

### #8012 - [Moore][Arc][LLHD] Moore to LLVM lowering issues
- **Similarity Score**: 6.0/15
- **State**: OPEN
- **Labels**: None
- **Matched Keywords**: arcilator, LLHD
- **URL**: https://github.com/llvm/circt/issues/8012
- **Analysis**: General Moore/Arc/LLHD pipeline issues. Different error (`llhd.process` not supported by ConvertToArcs). Same problem area but different root cause.

### #8286 - [circt-verilog][llhd][arcilator] Verilog-to-LLVM lowering issues
- **Similarity Score**: 5.5/15
- **State**: OPEN
- **Labels**: None
- **Matched Keywords**: arcilator, LLHD
- **URL**: https://github.com/llvm/circt/issues/8286
- **Analysis**: Collection of lowering issues with different errors (`llhd.constant_time`, `non-pure operation`). Not the same crash mechanism.

### #8065 - [LLHD][Arc] Indexing and slicing lowering from Verilog to LLVM IR
- **Similarity Score**: 4.0/15
- **State**: OPEN
- **Labels**: None
- **Matched Keywords**: arcilator, LLHD
- **URL**: https://github.com/llvm/circt/issues/8065
- **Analysis**: Indexing/slicing issues, different mechanism from inout port handling.

### #4916 - [Arc] LowerState: nested arc.state get pulled in wrong clock tree
- **Similarity Score**: 4.0/15
- **State**: OPEN
- **Labels**: Arc
- **Matched Keywords**: LowerState, Arc
- **URL**: https://github.com/llvm/circt/issues/4916
- **Analysis**: LowerState pass issue but about clock tree ordering, not type verification failure.

### #9395 - [circt-verilog][arcilator] Arcilator assertion failure
- **Similarity Score**: 3.5/15
- **State**: CLOSED
- **Labels**: Arc, ImportVerilog
- **Matched Keywords**: arcilator, assertion
- **URL**: https://github.com/llvm/circt/issues/9395
- **Analysis**: Different assertion failure in ConvertToArcs pass (DialectConversion.cpp), not StateType verification.

## Recommendation
**likely_new**

### Reasoning
1. **No exact duplicate found**: No existing issue reports the specific assertion failure `state type must have a known bit width; got '!llhd.ref<i8>'`

2. **Related tracking issue exists**: Issue #8825 tracks the underlying limitation (need for proper `llhd.ref` type), but:
   - It's a feature request, not a bug report
   - It doesn't mention the crash or assertion failure
   - Our report would add value by documenting the specific failure mode

3. **Distinct from similar issues**: Other arcilator/LLHD issues (#8012, #8286, #8065) have different error messages and crash locations

### Suggested Action
File a **new bug report** that:
- Describes the assertion failure when using inout ports with arcilator
- References #8825 as the related feature tracking issue
- Provides the minimal reproduction case
- Notes that the workaround is to avoid inout ports until #8825 is implemented
