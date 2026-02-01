# CIRCT Duplicate Issue Check Report

## Executive Summary

A search for duplicate issues related to the analyzed crash has been completed. **A highly likely duplicate issue was found.**

- **Top Match**: Issue #9574 - "[Arc] Assertion failure when lowering inout ports in sequential logic"
- **Similarity Score**: 7.0/10.0
- **Status**: OPEN
- **Recommendation**: **likely_duplicate**

---

## Analyzed Crash Details

### Error Information
- **Error Message**: `state type must have a known bit width; got '!llhd.ref<i1>'`
- **Tool**: arcilator
- **Pass**: arc-lower-state
- **Component**: Arc Dialect
- **Crash Location**: `lib/Dialect/Arc/Transforms/LowerState.cpp:219`

### Root Cause
The arcilator's `LowerState` pass attempts to create `StateType` for module input arguments that are LLHD reference types (used for inout ports). The `computeLLVMBitWidth()` function does not handle `llhd::RefType`, causing `StateType::verify()` to fail.

### Key Keywords
- StateType, state type, bit width
- llhd.ref, RefType
- arcilator, LowerState
- inout, bidirectional port

---

## Search Strategy

Multiple search queries were executed against the CIRCT repository:

1. `llhd.ref StateType` - Direct component matching
2. `arcilator LowerState` - Tool and pass matching
3. `RefType inout` - Type and port mechanism matching
4. `bit width state type` - Error message keywords
5. `bidirectional port` - Trigger condition
6. `known bit width` - Error constraint phrase

---

## Duplicate Detection Results

### Top Match: Issue #9574

**Title**: [Arc] Assertion failure when lowering inout ports in sequential logic  
**Status**: OPEN  
**Similarity Score**: 7.0/10.0

#### Matched Keywords (5/13)
- ✅ LowerState
- ✅ Arc (dialect)
- ✅ StateType
- ✅ inout
- ✅ assertion failure

#### Key Differences
The issue #9574 specifically mentions:
- SystemVerilog `always_ff` blocks with inout ports
- The exact same error message: "state type must have a known bit width; got '!llhd.ref<i1>'"
- The exact same crash location: `LowerState.cpp:219`
- Command: `circt-verilog --ir-hw bug.sv | arcilator`

#### Analysis
This is **almost certainly the exact same bug**. The crash signatures, error messages, and affected code are identical. The main difference is that issue #9574 provides a minimal reproduction case with `always_ff` blocks, while the test case may use a different SystemVerilog construct that triggers the same code path.

---

### Secondary Match: Issue #9417

**Title**: [Arc][arcilator] `hw.bitcast` Data Corruption for Aggregate Types  
**Status**: CLOSED (Fixed)  
**Similarity Score**: 4.0/10.0

#### Relevance
This is a different Arc/arcilator issue involving type handling, but the underlying cause (missing type support in Arc operations) is related.

---

## Conclusion

### Recommendation: **likely_duplicate**

Based on the analysis:
1. **Identical error message and crash location** - Strong indicator
2. **Same tool (arcilator) and pass (arc-lower-state)** - Specific component match
3. **Same dialect affected (Arc)** - Confirmed
4. **Same root cause**: Missing type support in computeLLVMBitWidth()

### Suggested Action

**Check if this test case reproduces the same issue as #9574.** If so:
1. Consider marking as duplicate of #9574
2. Reference the test case from this analysis in the discussion
3. If there's a subtle difference, document it for comprehensive fix validation

---

## Search Coverage

**Searches Executed**: 6  
**Results Retrieved**: Multiple  
**Relevant Issues Found**: 2  
**High Confidence Matches**: 1 (Issue #9574)

---

## Timestamp

Generated: 2024-12-20  
Repository: llvm/circt  
Dialect: Arc/LLHD  
Tool: arcilator
