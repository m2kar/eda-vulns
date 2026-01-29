# Validation Report

## Summary

| Check | Result |
|-------|--------|
| Syntax Check | ✅ valid |
| Feature Support | ✅ supported |
| Known Limitations | none |
| **Classification** | **report** |

## Test Case

```systemverilog
typedef union packed { logic a; } my_union;
module Sub(input my_union x);
endmodule
```

## Syntax Validation

**Tool**: slang
**Status**: ✅ Pass

```
Build succeeded: 0 errors, 0 warnings
```

The test case is valid IEEE 1800-2017 SystemVerilog. Packed unions are part of the standard (Section 7.3).

## Feature Support Analysis

**Unsupported features detected**: None

The test case uses only standard SystemVerilog features:
- `typedef` - type alias
- `union packed` - packed union type
- `module` with typed port

### CIRCT Known Limitations

No known limitation matched. This appears to be an implementation gap in the MooreToCore type conversion.

## Cross-Tool Validation

| Tool | Status | Exit Code | Notes |
|------|--------|-----------|-------|
| Slang | ✅ Pass | 0 | Build succeeded: 0 errors, 0 warnings |
| Verilator | ✅ Pass | 0 | No errors or warnings |
| Icarus Verilog | ✅ Pass | 0 | Compilation successful |

**Conclusion**: All three industry-standard tools accept this test case without errors.

## Classification

**Result**: `report`

**Confidence**: High

**Reasoning**:
The test case is valid SystemVerilog code that is accepted by all other major tools (Slang, Verilator, Icarus Verilog). The crash in CIRCT is due to missing type converter for `moore::UnionType` in the MooreToCore pass. This is a genuine bug that should be reported.

## Recommendation

**Proceed to check for duplicates and generate the bug report.**

The issue is:
- **Reproducible**: Yes, consistently crashes
- **Valid SV**: Yes, passes all syntax checks
- **Unique to CIRCT**: Yes, other tools handle it correctly
- **Root cause identified**: Missing UnionType converter in MooreToCore

## IEEE 1800-2017 Reference

Section 7.3 - Packed unions:
> A packed union is a union in which all members are of the same size. A packed union contains at least one member that is a packed data type.

The test case follows this specification correctly.
