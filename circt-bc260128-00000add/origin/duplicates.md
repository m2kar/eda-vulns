# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 78 |
| Top Similarity Score | 4.5 |
| **Recommendation** | **likely_new** |
| Confidence | medium |

## Search Parameters

| Parameter | Value |
|-----------|-------|
| Dialect | Arc |
| Failing Pass | arc::LowerStatePass |
| Crash Type | assertion |

## Keywords Used

- arcilator
- inout
- llhd.ref
- StateType
- LowerStatePass
- bit width
- bidirectional
- tristate
- computeLLVMBitWidth

## Top Similar Issues

### [#8212](https://github.com/llvm/circt/issues/8212) - Score: 4.5

**Title**: [arcilator] Install configuration missing for arcilator-runtime.h

**State**: OPEN

**Labels**: Arc

---

### [#8484](https://github.com/llvm/circt/issues/8484) - Score: 4.5

**Title**: [arcilator] Add functionality to the arcilator runtime library

**State**: OPEN

**Labels**: Arc

---

### [#9260](https://github.com/llvm/circt/issues/9260) - Score: 4.5

**Title**: Arcilator crashes in Upload Release Artifacts CI

**State**: OPEN

**Labels**: bug, Arc

---

### [#9467](https://github.com/llvm/circt/issues/9467) - Score: 4.5

**Title**: [circt-verilog][arcilator] `arcilator` fails to lower `llhd.constant_time` generated from simple SV delay (`#1`)

**State**: OPEN

**Labels**: LLHD, Arc

---

### [#8825](https://github.com/llvm/circt/issues/8825) - Score: 4.0

**Title**: [LLHD] Switch from hw.inout to a custom signal reference type

**State**: OPEN

**Labels**: LLHD

---

## Recommendation Details

📋 **Proceed with Caution**

Related issues exist but this appears to be a different bug.

**Recommended Action:**
- Proceed with generating the bug report
- Reference related issues in the report
- Highlight what makes this bug different


## Scoring Methodology

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
| Failing pass match | 2.0 | If failing pass appears in issue |
