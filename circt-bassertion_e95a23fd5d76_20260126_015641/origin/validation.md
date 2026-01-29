# Validation Report

## Summary

| Check | Result |
|-------|--------|
| Syntax Check | valid |
| Feature Support | supported |
| Known Limitations | none |
| **Classification** | **historical_bug** |

> **Note**: This is a historical bug report. The crash occurred in CIRCT version 1.139.0 but is now fixed in the current version.

## Syntax Validation

**Tool**: slang 10.0.6+3d7e6cd2e
**Status**: valid

```
Build succeeded: 0 errors, 0 warnings
```

## Feature Support Analysis

**Unsupported features detected**: None

### Features Used in Test Case

| Feature | IEEE Standard | CIRCT Support |
|---------|---------------|---------------|
| typedef | 1800-2005 | Supported |
| struct packed | 1800-2005 | Supported |
| logic | 1800-2005 | Supported |
| module | 1364/1800 | Supported |
| always_comb | 1800-2005 | Supported |
| assign | 1364/1800 | Supported |
| inout wire | 1364/1800 | Supported |

### CIRCT Known Limitations

No known limitation matched.

## Cross-Tool Validation

| Tool | Version | Status | Notes |
|------|---------|--------|-------|
| Verilator | 5.022 | pass | No errors or warnings |
| Icarus Verilog | 13.0 | pass | Compiled successfully |
| Slang | 10.0.6 | pass | Syntax check passed |

**Conclusion**: All three tools accept this test case as valid SystemVerilog.

## Classification

**Result**: `historical_bug`

**Reasoning**:
This test case is valid SystemVerilog code that was accepted by all validation tools. The original crash occurred in CIRCT version 1.139.0 in the arcilator tool's `LowerState` pass. The bug has been fixed in the current CIRCT version.

### Original Crash Details

- **Tool**: arcilator
- **Failing Pass**: LowerState
- **Crash Type**: Assertion failure
- **Assertion**: `state type must have a known bit width; got '!llhd.ref<i1>'`
- **Location**: `circt::arc::StateType::get` in `LowerState.cpp:219`

### Root Cause (Historical)

The arcilator's `LowerState` pass attempted to create a `StateType` with an LLHD reference type (`!llhd.ref<i1>`) which doesn't have a known bit width. The `StateType::get` function has a verifier that requires types with known bit widths, causing the assertion failure.

This likely occurred when processing the `inout wire c` port, which gets lowered to an LLHD reference type in the intermediate representation.

## Recommendation

**Action**: Document as historical bug for regression testing.

This test case can be added to CIRCT's regression test suite to ensure the fix remains in place. The crash was a genuine bug in the arcilator's handling of LLHD reference types during state lowering.

## Test Case

```systemverilog
typedef struct packed {
  logic [3:0] field1;
  logic valid;
} my_struct_t;

module MixedPorts(input logic a, output logic b, inout wire c);
  my_struct_t data;

  always_comb begin
    data.field1 = 4'b1100;
    data.valid = a;
  end

  assign b = data.valid;
endmodule
```

## Reproduction Command (Historical)

```bash
circt-verilog --ir-hw source.sv | arcilator
```

**Note**: This command no longer crashes on the current CIRCT version.
