# Validation Report

## Summary

| Check | Result |
|-------|--------|
| Syntax Check | ✅ valid |
| Feature Support | ✅ supported |
| Known Limitations | ✅ none matched |
| **Classification** | **report** |

## Syntax Validation

**Tool**: slang 10.0.6
**Status**: valid

```
Build succeeded: 0 errors, 0 warnings
```

## Feature Support Analysis

**Unsupported features detected**: None

The test case uses only basic SystemVerilog constructs:
- `struct packed` - Widely supported
- `always_comb` - IEEE 1800-2005
- Continuous assignment - Basic Verilog

### CIRCT Known Limitations

No known limitation matched.

## Cross-Tool Validation

| Tool | Status | Errors | Warnings | Notes |
|------|--------|--------|----------|-------|
| Slang 10.0.6 | ✅ pass | 0 | 0 | Clean |
| Verilator 5.022 | ⚠️ warning | 0 | 3 | Lint warnings only (DECLFILENAME, UNUSEDSIGNAL, UNDRIVEN) |
| Icarus 13.0 | ✅ pass | 0 | 1 | Benign sensitivity warning |

### Verilator Output
```
%Warning-DECLFILENAME: Filename 'bug' does not match MODULE name: 'M'
%Warning-UNUSEDSIGNAL: Bits of signal are not used: 's'[1]
%Warning-UNDRIVEN: Bits of signal are not driven: 's'[0]
```
These are lint warnings, not syntax errors. The code is valid.

### Icarus Output
```
bug.sv:3: warning: always_comb process has no sensitivities.
```
This is a benign warning about always_comb with a constant assignment.

## Classification

**Result**: `report`

**Reasoning**:
The test case is valid SystemVerilog that is accepted by all three major validation tools (slang, verilator, icarus). CIRCT's `circt-verilog --ir-hw` command times out when processing this code, indicating a bug in the MooreToHW conversion pass.

## Minimization Verification

| Metric | Value |
|--------|-------|
| Original lines | 35 |
| Minimized lines | 5 |
| Reduction | 85.7% |
| Minimal | ✅ Yes |

All components verified as essential:
- `always_comb` assigning struct field → Required (assign does not trigger)
- Reading different struct field → Required
- Output port using read value → Required (prevents dead code elimination)

## Bug Pattern

**Trigger**: Partial struct field assignment in `always_comb` combined with reading a different field of the same struct.

**Minimal Reproducer**:
```systemverilog
module M (output logic z);
  struct packed { logic a; logic b; } s;
  always_comb s.a = 1;
  assign z = s.b;
endmodule
```

## Recommendation

**Proceed to check for duplicates and generate the bug report.**

This is a valid compiler bug that causes CIRCT to hang during MooreToHW conversion. The pattern involves:
1. Packed struct with multiple fields
2. `always_comb` assigning to one field
3. Continuous assignment reading a different field

The bug likely relates to how partial struct assignments in procedural blocks are combined with field extractions in the HW dialect lowering.
