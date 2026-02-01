# Root Cause Analysis Report

## Executive Summary

The arcilator tool fails to legalize the `sim.fmt.literal` operation generated from a SystemVerilog concurrent assertion using `assert ... else $error(...)`. This is a known feature gap: Arcilator lacks support for assertion-related operations from the Sim dialect, specifically the formatted string literal operations used for error messages.

## Crash Context

- **Tool/Command**: `circt-verilog --ir-hw | arcilator`
- **Dialect**: Sim (for formatted strings), Arc (target)
- **Failing Pass**: Legalization pass in arcilator
- **Crash Type**: Legalization Failure (not assertion/segfault)

## Error Analysis

### Error Message
```
<stdin>:3:10: error: failed to legalize operation 'sim.fmt.literal'
    %0 = sim.fmt.literal "Error: Assertion failed: arr["
         ^
<stdin>:3:10: note: see current operation: %301 = "sim.fmt.literal"() <{literal = "Error: Assertion failed: arr["}> : () -> !sim.fstring
```

### Key Observations
1. The error occurs during arcilator's conversion/legalization pass
2. The `sim.fmt.literal` operation produces an `!sim.fstring` type
3. This operation comes from lowering the SystemVerilog `$error()` system task
4. Arcilator has no conversion pattern for `sim.fmt.literal`

## Test Case Analysis

### Code Summary
The test case defines a simple module with:
- Clock and reset inputs
- An 8-bit counter that increments on each clock cycle
- A 256-element array `arr`
- A concurrent assertion in `always_comb` that checks `arr[counter] == 1'b1`

### Key Constructs
- **`always_comb` block**: Concurrent procedural block
- **`assert ... else $error()`**: Concurrent assertion with formatted error message
- **Array indexing**: `arr[counter]` - dynamic array access with runtime index

### Problematic Pattern
```systemverilog
always_comb begin
  assert (arr[counter] == 1'b1) else $error("Assertion failed: arr[%0d] != 1", counter);
end
```

The `$error()` system task with a format string is converted to `sim.fmt.literal` and `sim.fmt.*` operations in the IR. While the SimToSV conversion handles these for Verilog output, there is no SimToArc or equivalent lowering for arcilator.

## CIRCT Source Analysis

### Conversion Pipeline
The tool chain executes:
1. `circt-verilog --ir-hw`: Parses SV and emits HW dialect IR
2. `arcilator`: Should lower to Arc dialect then to LLVM

### Missing Conversion
- **SimToSV exists**: `lib/Conversion/SimToSV/SimToSV.cpp` - handles `sim.*` ops for Verilog output
- **SimToArc missing**: There is no `lib/Conversion/SimToArc/` directory
- The Sim dialect operations (`sim.fmt.literal`, `sim.fmt.concat`, etc.) have no lowering path to Arc/LLVM

### Processing Path
1. `circt-verilog` parses SV and generates:
   - `sim.fmt.literal "Error: Assertion failed: arr["` for the string literal
   - Additional `sim.fmt.*` ops for the format specifier and argument
2. Arcilator runs legalization with target dialects (Arc, LLVM, etc.)
3. Legalization fails because `sim.fmt.literal` has no conversion pattern
4. Error is emitted and compilation aborts

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: Arcilator lacks lowering support for Sim dialect assertion and formatting operations

**Evidence**:
- GitHub Issue #6810 "[Arc] Add basic assertion support" is open and marked "good first issue"
- No `SimToArc` conversion exists in the codebase
- `SimToSV` exists but only targets SV dialect output, not Arc/LLVM

**Mechanism**: 
- The `$error()` system task in SV is converted to `sim.fmt.*` operations
- Arcilator's legalization pass has no patterns to convert these operations
- MLIR's conversion framework reports the operation as illegal

### Hypothesis 2 (Medium Confidence)
**Cause**: The assertion feature is intentionally not supported yet in arcilator

**Evidence**:
- Issue #6810 suggests this is a planned feature, not a bug
- The issue is tagged with "good first issue", indicating it's a known gap

**Mechanism**:
- Arcilator focuses on simulation of combinational and sequential logic
- Assertion handling requires runtime support for formatted output
- This feature has simply not been prioritized/implemented yet

## Suggested Fix Directions

1. **Implement SimToArc conversion** (or SimToLLVM through Arc):
   - Add lowering patterns for `sim.fmt.literal`, `sim.fmt.concat`, etc.
   - Map to appropriate Arc or LLVM operations for runtime assertion support

2. **Strip assertions before arcilator** (workaround):
   - Add a pass to remove or no-op assertion-related operations
   - Would allow simulation to proceed without assertion checks

3. **Improve error message**:
   - Emit a clear diagnostic that assertions are not supported in arcilator
   - Suggest using `--strip-assertions` or similar flag (if implemented)

## Keywords for Issue Search
`sim.fmt.literal` `arcilator` `assertion` `legalize` `sim dialect` `$error` `fstring`

## Related Files to Investigate
- `lib/Conversion/SimToSV/SimToSV.cpp` - Reference implementation for Sim dialect lowering
- `tools/arcilator/arcilator.cpp` - Arcilator tool implementation
- `lib/Dialect/Sim/SimOps.cpp` - Sim dialect operation definitions
- `include/circt/Dialect/Sim/SimOps.td` - Sim dialect TableGen definitions

## Related Issues
- **Issue #6810**: "[Arc] Add basic assertion support" - Open, good first issue
- This crash is a manifestation of the missing feature described in #6810
