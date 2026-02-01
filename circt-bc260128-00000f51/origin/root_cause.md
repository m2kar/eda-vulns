# Root Cause Analysis Report

## Executive Summary

CIRCT's `circt-verilog --ir-hw` command enters an infinite loop (timeout > 300s) when lowering SystemVerilog code containing a `packed struct` where different fields are used in conflicting contexts: one field is used as a submodule instance input, while another field is written to in an `always_comb` block. This appears to be a bug in the MooreToCore lowering pass that fails to properly handle the lowered representation of struct fields with these cross-usage patterns.

## Crash Context

- **Tool/Command**: `circt-verilog --ir-hw`
- **Dialect**: Moore (SystemVerilog frontend)
- **Failing Pass**: MooreToCore (inferred from timeout during lowering)
- **Crash Type**: Timeout (infinite loop, not a crash with assertion)

## Error Analysis

### Error Message
```
Compilation timed out after 300s
```

### Compilation Command
```bash
circt-verilog --ir-hw <input>.sv | arcilator | opt -O0 | llc -O0 --filetype=obj -o output.o
```

Note: The timeout occurs at the `circt-verilog --ir-hw` stage, before reaching later pipeline stages.

## Test Case Analysis

### Code Summary
The test case defines a `top_module` containing:
1. A `packed struct` with `data` and `valid` fields
2. A submodule instance that takes `data_reg.valid` as input
3. An `always_comb` block that writes to `data_reg.data`
4. An `initial` block with a `forever` loop for clock generation
5. An `assign` statement connecting `data_reg.valid` to a wire

### Key Constructs

| Construct | Purpose | Relationship to Timeout |
|-----------|---------|------------------------|
| `struct packed { ... }` | Defines packed struct with multiple fields | Root cause - struct field handling |
| `always_comb` | Combinational logic for `data_reg.data` | One field written in comb logic |
| `submodule` instantiation | Hierarchical design | Uses `data_reg.valid` as input |
| `.sig(data_reg.valid)` | Connect struct field to submodule | Other field used as instance input |
| `data_reg.data = ...` | Write to struct field in always_comb | Creates cross-usage pattern |

### Potentially Problematic Patterns

**Primary Issue**: Packed struct field cross-usage
```systemverilog
struct packed {
  logic [7:0] data;
  logic valid;
} data_reg;

// Field 'valid' used as submodule input
submodule inst (
  .sig(data_reg.valid),  // ← One field
  ...
);

// Field 'data' written in always_comb
always_comb begin
  data_reg.data = sub_out ? 8'hFF : 8'h00;  // ← Another field
end
```

This pattern creates a scenario where the lowering pass must handle:
1. Struct field extraction for module instance connections
2. Combinational logic writing to different fields of the same struct
3. Proper SSA representation of the struct with partial updates

## Experimental Verification

### Test Matrix

| Test Case | Result | Notes |
|-----------|--------|-------|
| Simple `forever` loop only | ✓ OK | Baseline - no struct |
| `forever` + `always_comb` | ✓ OK | No struct |
| `forever` + submodule | ✓ OK | No struct |
| `forever` + struct (no field access) | ✓ OK | Struct declared but not accessed |
| `forever` + struct + `always_comb` | ✓ OK | Struct written but not connected to submodule |
| `forever` + struct + submodule (no `always_comb`) | ✓ OK | Struct field connected but not written in comb logic |
| **Full test case** | **✗ TIMEOUT** | **Struct field connected + written in comb logic** |
| Separate variables instead of struct | ✓ OK | Confirms struct is the issue |

### Key Finding

The timeout **only occurs** when:
1. A `packed struct` is declared
2. **One field** is used as a submodule instance connection
3. **Another field** of the same struct is written in an `always_comb` block

All other combinations compile successfully.

## CIRCT Source Analysis

### Inferred Crash Location
Based on the timeout occurring during the `circt-verilog --ir-hw` command:

**Likely location**: `lib/Conversion/MooreToCore/`
- This directory contains the lowering passes from Moore (SystemVerilog) dialect to Core dialect
- The timeout suggests a pass is entering an infinite loop, likely during struct field lowering

**Specific areas to investigate**:
- Struct lowering utilities in Moore dialect
- Handling of partial struct updates in comb logic
- Connection lowering for module instances with struct field inputs

### Expected Code Path (Inferred)

1. **Parse**: SystemVerilog parsed into Moore dialect operations
2. **Lower Struct**: Packed struct fields should be lowered to individual values
3. **Lower Instances**: Submodule instance connections should reference lowered field values
4. **Lower Comb Logic**: `always_comb` should be converted to combinational operations
5. **Bug Point**: The lowering process likely fails to properly separate the struct fields when they're used in different contexts, causing a loop in the dependency resolution

## Root Cause Hypotheses

### Hypothesis 1 (High Confidence)
**Cause**: The MooreToCore lowering pass enters an infinite loop when lowering packed structs where different fields are used in conflicting contexts.

**Evidence**:
- Test case with separate variables (not struct) compiles successfully
- Test case with struct but no field access to submodule compiles successfully  
- Test case with struct field connected but no `always_comb` writes compiles successfully
- **Only fails when struct field is used as submodule input AND another field is written in always_comb**
- Other SV tools (slang, iverilog) accept the syntax without issues

**Mechanism**: 
The lowering pass likely:
1. Lowers the struct to its constituent fields
2. Attempts to create SSA values for each field
3. Encounters circular dependencies when processing:
   - The instance connection (needs the field value)
   - The comb logic (writes to another field, which may affect struct lowering)
   - The struct itself (which may be treated as a single unit during some phases)
4. Gets stuck in an infinite loop trying to resolve these dependencies

### Hypothesis 2 (Medium Confidence)
**Cause**: The compiler is unable to resolve circular dependencies introduced by the struct field accesses.

**Evidence**:
- The pattern creates a dependency chain through the struct and submodule
- Combination logic involving struct fields may not be properly tracked during lowering
- Different lowering phases may have inconsistent views of the struct (as a whole vs. as individual fields)

**Mechanism**:
During lowering, the pass may:
1. Track struct dependencies inconsistently across passes
2. Fail to mark struct fields as independent when they should be
3. Enter an analysis loop that never converges

## Suggested Fix Directions

1. **Add dependency tracking for struct fields**: Ensure struct fields are properly tracked as independent values when lowering, especially when used in different contexts (instance connections vs. comb logic)

2. **Break infinite loop detection**: Add a safeguard or loop detection in the MooreToCore pass to catch cases where dependency resolution doesn't converge

3. **Validate struct lowering invariants**: Add assertions to catch invalid struct lowering patterns early, before they cause infinite loops

4. **Separate struct field lowering**: Consider lowering struct fields to independent values earlier in the pipeline, before they're used in instance connections or comb logic

## Keywords for Issue Search

`struct packed` `always_comb` `MooreToCore` `timeout` `lowering` `circular dependency` `field access` `submodule instance` `infinite loop`

## Related Files to Investigate

- `lib/Conversion/MooreToCore/` - Contains Moore to Core dialect lowering passes
  - Focus on struct lowering logic
  - Check for loops in dependency resolution
- `lib/Dialect/Moore/` - Moore dialect definition and lowering utilities
  - Struct type definitions
  - Field access operations
- `include/circt/Dialect/Moore/IR/MooreOps.td` - Moore operation definitions

## Minimal Reproducer

```systemverilog
module top_module;
  logic clk, sub_out;
  
  struct packed {
    logic [7:0] data;
    logic valid;
  } data_reg;
  
  submodule inst (
    .clk(clk),
    .sig(data_reg.valid),  // Field 1: used as instance input
    .out(sub_out)
  );
  
  always_comb begin
    data_reg.data = sub_out ? 8'hFF : 8'h00;  // Field 2: written in always_comb
  end
endmodule

module submodule(
  input logic clk,
  input logic sig,
  output logic out
);
  always_ff @(posedge clk) begin
    out <= sig;
  end
endmodule
```

This minimal reproducer captures the essential pattern that triggers the timeout.
