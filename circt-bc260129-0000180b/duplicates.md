# Duplicate Check Report

## Summary

| Metric | Value |
|--------|-------|
| Issues Found | 10 |
| Top Similarity Score | 0.0 |
| **Recommendation** | **new_issue** |

## Search Parameters

- **Dialect**: Moore
- **Failing Pass**: HoistSignals
- **Crash Type**: assertion
- **Keywords**: self-referential typedef, circular type, recursive type, parameterized class, HoistSignals, hw.bitcast, bitwidth overflow

## Top Similar Issues

### [Issue #6866](https://github.com/llvm/circt/issues/6866) (Score: 0.0)
**Title**: [OM] parsing depend on ordering of Class

**State**: OPEN

**Labels**: OM

**Relation**: About class type resolution, but different issue (ordering vs self-reference)

---

### [Issue #9287](https://github.com/llvm/circt/issues/9287) (Score: 0.0)
**Title**: [HW] Make `hw::getBitWidth` use std::optional vs -1

**State**: OPEN

**Labels**: HW

**Relation**: About getBitWidth API, but not about self-referential types

---

### [Issue #9570](https://github.com/llvm/circt/issues/9570) (Score: 0.0)
**Title**: [Moore] Assertion in MooreToCore when module uses packed union type as port

**State**: OPEN

**Labels**: bug, Moore

**Relation**: About union types as ports, different issue

---

### [Issue #2590](https://github.com/llvm/circt/issues/2590) (Score: 0.0)
**Title**: [ExportVerilog] Incorrect verilog output for bitcast + zero width aggregate types

**State**: OPEN

**Labels**: ExportVerilog

**Relation**: About bitcast, but for zero-width types, not overflow

---

### [Issue #2593](https://github.com/llvm/circt/issues/2593) (Score: 0.0)
**Title**: [ExportVerilog] Omit bitwidth of constant array index

**State**: OPEN

**Labels**: ExportVerilog

**Relation**: About bitwidth issues, but for array indices

---

### [Issue #8286](https://github.com/llvm/circt/issues/8286) (Score: 0.0)
**Title**: [circt-verilog][llhd][arcilator] Verilog-to-LLVM lowering issues

**State**: OPEN

**Labels**: (none)

**Relation**: General LLHD issue, not specific to our bug

---

### Additional Issues Found
- #6740 - Conversion failure of invalidated wire of clock type
- #3768 - Enum aliasing in SV output
- #2583 - LegalizeNames: Typedefs need to be checked if they collide with reserved words

## Recommendation

**Action**: `new_issue`

✅ **Clear to Proceed**

No similar issues were found. This is likely a new bug.

**Recommended:**
- Proceed to generate and submit bug report
- Reference related issues (#9287 about getBitWidth API, #9570 about Moore type handling) in your report
- Highlight that this is specifically about self-referential typedefs in parameterized classes

## Search Methodology

### Searches Performed
1. Keyword search: "self-referential typedef", "circular type", "recursive type", "parameterized class"
2. Assertion message search: "bitwidth in hardware is known"
3. Specific value search: "1073741823"
4. Pass-specific search: "HoistSignals"
5. Dialect-specific search: "Moore"
6. Related issue: #6866 (class type ordering)

### Analysis
None of the existing issues directly match:
- Self-referential typedef causing bit width overflow
- The specific error message about `hw.bitcast` and `i1073741823`
- The interaction between delay statements and self-referential types
- HoistSignals pass assertion failures related to circular types

### Related But Different Issues
- #9287: About class ordering, not self-reference
- #9287: About getBitWidth returning -1 vs optional
- #9570: Moore dialect assertion with union types
- Several bitcast/bitwidth-related issues in ExportVerilog

## Scoring Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Title keyword match | 2.0 | Per keyword found in title |
| Body keyword match | 1.0 | Per keyword found in body |
| Assertion message match | 3.0 | If assertion appears in body |
| Dialect label match | 1.5 | If dialect label matches |
| Failing pass match | 2.0 | If failing pass appears in issue |

## Conclusion

**This is a new bug** that should be filed as a fresh issue. While there are related issues in CIRCT (especially #9287 about getBitWidth and #9570 about Moore type assertions), none describe the specific interaction between self-referential typedefs and delay statements that triggers the bit width overflow error.
