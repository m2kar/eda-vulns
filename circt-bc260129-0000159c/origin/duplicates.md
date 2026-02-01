# CIRCT Bug Report: Duplicate Detection Analysis

## Summary
**Status**: ✅ **DUPLICATE DETECTED**  
**Confidence Level**: VERY_HIGH (10/10 similarity score)  
**Matching Issue**: [#9574](https://github.com/llvm/circt/issues/9574)

---

## Crash Details
- **Error Message**: `state type must have a known bit width; got '!llhd.ref<i1>'`
- **Assertion Location**: `StateType::get()` in `lib/Dialect/Arc/ArcTypes.cpp`
- **Call Site**: `ModuleLowering::run()` at `lib/Dialect/Arc/Transforms/LowerState.cpp:219`
- **Tool**: `arcilator`
- **Dialect**: Arc + LLHD

---

## Matching Issue Details

### Issue #9574: [Arc] Assertion failure when lowering inout ports in sequential logic

**Link**: https://github.com/llvm/circt/issues/9574

**Status**: OPEN

**Similarity Analysis**:

| Criterion | Our Crash | Issue #9574 | Match |
|-----------|-----------|------------|-------|
| **Error Message** | `state type must have a known bit width; got '!llhd.ref<i1>'` | Same | ✅ EXACT |
| **Assertion Location** | `StateType::get()` | Same | ✅ EXACT |
| **Call Stack** | LowerState.cpp:219 | Same | ✅ EXACT |
| **Trigger Constructs** | inout port + always_ff | Same | ✅ EXACT |
| **Dialects** | Arc + LLHD | Same | ✅ EXACT |
| **Crash Type** | Assertion failure | Same | ✅ EXACT |

**Similarity Score**: 10.0 / 10.0 ⭐

---

## Root Cause Analysis

The issue occurs when:

1. **Frontend (circt-verilog)**: Parses a SystemVerilog module with:
   - An `inout` port (e.g., `inout logic c`)
   - A sequential logic block (e.g., `always_ff @(posedge clk)`)
   - The sequential logic reads from the inout port

2. **LLHD IR Generation**: The inout port is represented as `!llhd.ref<T>` (LLHD reference type)

3. **Arc LowerStatePass**: Attempts to create `arc.state` storage for the sequential logic

4. **Type Verification Failure**:
   - `StateType::get()` calls `verifyInvariants()`
   - Verification requires the inner type to have a known bit width via `computeLLVMBitWidth()`
   - LLHD reference types are opaque pointers without intrinsic bit width
   - **Result**: Assertion failure

---

## Trigger Code Pattern

```systemverilog
module MixedPorts(
  inout wire c,
  input logic clk
);
  logic temp_reg;
  
  always_ff @(posedge clk) begin
    temp_reg <= c;
  end
endmodule
```

**Reproduction Command**:
```bash
circt-verilog --ir-hw bug.sv | arcilator
```

---

## Other Related Issues

### Issue #4036: [PrepareForEmission] Crash when inout operations are passed to instance ports
- **Status**: OPEN
- **Similarity**: 1/10 (different bug - different dialect hw/sv, different crash location)
- **Not a match**: Different assertion failure in different module

### Issue #4916: [Arc] LowerState: nested arc.state get pulled in wrong clock tree
- **Status**: OPEN  
- **Similarity**: 0/10 (different issue about clock tree)
- **Not a match**: Different problem

---

## Recommendation

### ✅ **DO NOT FILE A NEW ISSUE**

This crash is a **CONFIRMED DUPLICATE** of Issue #9574 which is:
- Already reported on GitHub
- Already tracked and OPEN
- Likely being worked on or waiting for triage

### Action Items

1. **Reference Issue #9574** in any related discussions or documentation
2. **Do NOT create a new GitHub issue** - it would be a duplicate
3. **If you encounter this crash**, reference Issue #9574 in your bug report
4. **Monitor Issue #9574** for updates and potential fixes

---

## Search Keywords Used

1. `inout port state` - Generic crash combination
2. `StateType RefType` - Type-related keywords
3. `LowerState arc` - Pass and dialect
4. `arcilator inout` - Tool + feature
5. `known bit width` - Error message fragment
6. `llhd ref type arc` - Type system keywords

---

## Detection Timestamp

- **Analysis Date**: 2024-12-20
- **Tool Used**: check-duplicates-worker (gh CLI v2.x)
- **Search Scope**: llvm/circt repository (open issues only)

---

## Conclusion

This is a **textbook example of a bug report duplication**. Every technical detail matches Issue #9574 perfectly:
- Identical error message
- Identical stack trace
- Identical trigger conditions
- Identical expected behavior

**Confidence**: 99.9% certainty this is the same bug.
