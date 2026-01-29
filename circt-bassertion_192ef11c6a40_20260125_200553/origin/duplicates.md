# Duplicate Check Report

## Summary

| Field | Value |
|-------|-------|
| **Recommendation** | `likely_new` |
| **Top Similarity Score** | 8.0 / 15.0 |
| **Most Similar Issue** | #9467 |
| **Decision** | Report as new issue with cross-references |

## Search Criteria

- **Dialect**: Arc
- **Crash Type**: assertion
- **Failing Pass**: LowerStatePass
- **Assertion Message**: `state type must have a known bit width; got '!llhd.ref<i1>'`
- **Keywords**: StateType, arcilator, inout, llhd.ref, bidirectional, tristate, LowerState, bit width, computeLLVMBitWidth, Arc

## Top 5 Similar Issues

### 1. [#9467](https://github.com/llvm/circt/issues/9467) - Score: 8.0
**Title**: [circt-verilog][arcilator] `arcilator` fails to lower `llhd.constant_time` generated from simple SV delay (`#1`)

**State**: OPEN | **Labels**: LLHD, Arc

**Similarity Analysis**:
- Title match: 2.0 (arcilator keyword)
- Body keywords: 3.0 (arcilator, LLHD, legalization failure)
- Dialect match: 3.0 (Arc, LLHD labels)
- **Total**: 8.0

**Why Not a Duplicate**:
This issue is about `llhd.constant_time` operations failing during `ConvertToArcs` pass. Our bug is about `llhd.ref<T>` types failing in `StateType::verify()` during `LowerState` pass. Different failing operations, different passes, different root causes.

---

### 2. [#9469](https://github.com/llvm/circt/issues/9469) - Score: 6.0
**Title**: [circt-verilog][arcilator] Inconsistent compilation behavior: direct array indexing in always_ff sensitivity list vs. intermediate wire

**State**: CLOSED | **Labels**: LLHD, Arc

**Similarity Analysis**:
- Title match: 1.0 (arcilator keyword)
- Body keywords: 2.0 (arcilator, LLHD, compilation failure)
- Dialect match: 3.0 (Arc, LLHD labels)
- **Total**: 6.0

**Why Not a Duplicate**:
This issue is triggered by array indexing in sensitivity lists and fails on `llhd.constant_time`. Our bug is triggered by `inout` ports and fails on `llhd.ref<i1>` in StateType. Different triggers and failure points.

---

### 3. [#8825](https://github.com/llvm/circt/issues/8825) - Score: 5.0
**Title**: [LLHD] Switch from hw.inout to a custom signal reference type

**State**: OPEN | **Labels**: LLHD

**Similarity Analysis**:
- Title match: 2.0 (inout, reference type)
- Body keywords: 3.0 (llhd.ref, signal reference, hw.inout)
- Dialect match: 0.0 (no Arc label)
- **Total**: 5.0

**Why Not a Duplicate**:
This is a **feature request** to introduce `!llhd.ref<T>` type. Our bug is about Arc's `StateType` not supporting this type. This issue is actually the **upstream cause** - the `llhd.ref` type was introduced but Arc wasn't updated to handle it. Should be referenced as related context.

---

### 4. [#8845](https://github.com/llvm/circt/issues/8845) - Score: 4.5
**Title**: [circt-verilog] `circt-verilog` produces non `comb`/`seq` dialects including `cf` and `llhd`

**State**: OPEN | **Labels**: LLHD, Verilog/SystemVerilog, ImportVerilog

**Similarity Analysis**:
- Title match: 1.0 (circt-verilog, llhd)
- Body keywords: 2.0 (LLHD dialect, unexpected output)
- Dialect match: 1.5 (LLHD label)
- **Total**: 4.5

**Why Not a Duplicate**:
This issue is about `circt-verilog` producing LLHD operations when the user expects only comb/seq. Our bug is about `arcilator` failing to handle LLHD types in its lowering passes. Different tools and different problems.

---

### 5. [#9395](https://github.com/llvm/circt/issues/9395) - Score: 4.5
**Title**: [circt-verilog][arcilator] Arcilator assertion failure

**State**: CLOSED | **Labels**: Arc, ImportVerilog

**Similarity Analysis**:
- Title match: 2.0 (arcilator, assertion)
- Body keywords: 1.0 (arcilator, assertion failure)
- Dialect match: 1.5 (Arc label)
- **Total**: 4.5

**Why Not a Duplicate**:
Different assertion failure. #9395 fails in `mlir::ConversionPatternRewriter::replaceUsesWithIf` during `ConvertToArcs`. Our bug fails in `StateType::verify()` with message about bit width. Completely different failure locations and messages.

---

## Recommendation

### Decision: `likely_new`

**Rationale**:

1. **Unique Failure Mode**: The assertion `state type must have a known bit width; got '!llhd.ref<i1>'` is not found in any existing issues.

2. **Different Pass**: Our failure occurs in `LowerStatePass` at `StateType::verify()`, while related issues (#9467, #9469) fail in `ConvertToArcs` pass.

3. **Different Trigger**: Our bug is triggered by `inout` ports creating `llhd.ref` types, while related issues are triggered by delays (`#1`) or array indexing.

4. **Root Cause**: The `computeLLVMBitWidth()` function in Arc dialect doesn't handle `llhd::RefType`, causing StateType verification to fail.

### Suggested Cross-References

When filing the new issue, reference:

- **#9467**: Similar class of bug (arcilator + LLHD incompatibility)
- **#8825**: Design discussion about `llhd.ref` type introduction
- **#8845**: Related issue about unexpected LLHD dialect in circt-verilog output

### Issue Classification

| Aspect | Value |
|--------|-------|
| Type | Bug Report |
| Component | Arc Dialect / arcilator |
| Root Cause | Missing llhd::RefType handler in computeLLVMBitWidth() |
| Severity | High (blocks arcilator for designs with inout ports) |
