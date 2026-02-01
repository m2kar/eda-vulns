# CIRCT Duplicate Bug Report Check

## Summary
Checked CIRCT repository for duplicate bug reports related to `sim.fmt.literal` legalization failure in the ArcToLLVM pass.

**Recommendation**: **new_issue** (Low similarity to existing issues)
- Top similarity score: 4.8/10
- Top related issue: #8012

---

## Search Strategy

### Primary Keywords
- `sim.fmt.literal` - The specific operation that crashes
- `legalization` - The type of failure
- `assertion` - The context ($error in assertions)
- `$error` - SystemVerilog system task

### Secondary Keywords
- `FormatLiteralOp` - Operation class name
- `ArcToLLVM` - Affected pass
- `arcilator` - Tool chain component
- `format string` - Related concept
- `combinational always` - Trigger context
- `SimDialect` - Related dialect

### Search Queries Executed
1. `sim.fmt.literal` - No results
2. `legalization` - 10 results (mostly HW/FIRRTL related)
3. `ArcToLLVM` - 3 results (unrelated to this issue)
4. `FormatLiteralOp` - No results
5. `arcilator` - 30+ results
6. `assertion` - 40+ results
7. `format string legalization` - No results
8. `error assertion verilog` - Various results (mostly unrelated)

---

## Found Issues Analysis

### Issue #8012 - [Moore][Arc][LLHD] Moore to LLVM lowering issues
- **State**: OPEN
- **Similarity Score**: 4.8/10
- **URL**: https://github.com/llvm/circt/issues/8012
- **Key Overlaps**: `arcilator`

**Analysis**:
- Reports generic LLVM lowering issues in arcilator
- Errors: `'llhd.process' op has regions; not supported by ConvertToArcs` and `'seq.clock_inv' legalization`
- Different from current bug (missing sim.fmt.literal pattern)
- User asks about DFF simulation, not assertions
- **Verdict**: Related but different root cause

---

### Issue #9467 - [circt-verilog][arcilator] arcilator fails to lower llhd.constant_time
- **State**: OPEN
- **Similarity Score**: 2.8/10
- **URL**: https://github.com/llvm/circt/issues/9467
- **Key Overlaps**: `arcilator`

**Analysis**:
- Also reports missing legalization pattern (for `llhd.constant_time`)
- Similar pattern: Missing conversion in ConvertToArcs pass
- Different operation: `llhd.constant_time` vs `sim.fmt.literal`
- Different trigger: SV `#1` delay vs assertion with $error
- **Verdict**: Similar pattern but different operation

---

### Issue #9395 - [circt-verilog][arcilator] Arcilator assertion failure
- **State**: CLOSED
- **Similarity Score**: 1.6/10
- **URL**: https://github.com/llvm/circt/issues/9395
- **Key Overlaps**: `assertion`, `arcilator`

**Analysis**:
- Already closed/resolved
- Generic "assertion failure" - no details on root cause
- Mentions assertion but in different context
- **Verdict**: Closed, unrelated specifics

---

### Issue #6810 - [Arc] Add basic assertion support
- **State**: OPEN
- **Similarity Score**: 0.8/10
- **URL**: https://github.com/llvm/circt/issues/6810
- **Key Overlaps**: `assertion`

**Analysis**:
- Feature request, not bug report
- Requests addition of `verif.assert` and `sv.assert.concurrent` support
- Generic assertion support, not specific legalization failure
- **Verdict**: Different issue type (enhancement vs bug)

---

## Detailed Bug Characteristics (Current Issue)

### Operation
- `sim.fmt.literal` - Format literal operation from SimDialect

### Root Cause
- Missing conversion pattern for standalone `sim.fmt.literal` in ArcToLLVM pass
- Operation is generated from `$error` system task in assertions
- Lacks proper lowering when not consumed by `sim.proc.print`

### Location
- File: `lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp`
- Pass: ArcToLLVM (part of arcilator pipeline)

### Trigger Pattern
- Assertion with `$error` in combinational always block
- Pattern: `always @(*) begin ... assert (...) else $error(...); end`

### Affected Passes
- ImportVerilog (generates the operation)
- ArcToLLVM (fails to legalize)

### Severity
- Error (crash during legalization)
- Reproducible: Yes

---

## Conclusion

The current bug report describes a **specific and unique issue**:

1. **Specific Operation**: `sim.fmt.literal` - No other issues mention this operation
2. **Specific Trigger**: Assertion with `$error` in combinational always blocks
3. **Specific Location**: ArcToLLVM pass legalization failure
4. **Root Cause**: Missing pattern for standalone format literal operations

While some existing issues describe similar patterns (missing legalization patterns in ConvertToArcs), none address the specific `sim.fmt.literal` operation or the assertion `$error` context.

### Recommendation
**Create a NEW ISSUE** - This bug has not been reported before in the CIRCT repository.

---

## Metadata
- Analysis Date: 2026-02-01
- Search Scope: llvm/circt repository
- Total Issues Reviewed: 4 (out of 80+ found in searches)
- Duplicate Risk: LOW (< 5.0/10 similarity)
