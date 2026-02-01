# Root Cause Analysis Report

## Crash ID: circt-bc260128-00000857

## Summary

The crash occurs when arcilator attempts to lower a `sim.fmt.literal` operation to LLVM IR. The operation is generated from a SystemVerilog `$error()` message in an immediate assertion statement. While the `sim.fmt.literal` operation is marked as "legal" in the LowerArcToLLVM pass (to be preserved for use in other lowering patterns), it is never actually consumed, leading to a legalization failure.

## Crash Details

- **Crash Type**: Legalization Failure (assertion)
- **Failed Operation**: `sim.fmt.literal`
- **Dialect**: Sim
- **Error Message**: `failed to legalize operation 'sim.fmt.literal'`

## Test Case Analysis

### Source Code (source.sv)

```systemverilog
module M(
  input logic enable,
  input logic [7:0] data_in,
  output logic data_out
);
  typedef struct packed {
    logic valid;
    logic [7:0] value;
  } reg_t;
  
  reg_t my_reg;
  logic q;
  assign q = my_reg.value[0];
  
  always_comb begin
    my_reg.valid = enable;
    my_reg.value = data_in;
  end
  
  assign data_out = my_reg.valid & my_reg.value[0];
  
  // Immediate assertion with $error message - TRIGGERS THE BUG
  always @(*) begin
    assert (q == 1'b0) else $error("Assertion failed: q != 0");
  end
endmodule
```

### Language Features Used

1. **Packed struct declaration** (`typedef struct packed`)
2. **Combinational logic** (`always_comb`, `assign`)
3. **Immediate assertion** (`assert ... else $error(...)`)
4. **$error system task** with literal string message

### Trigger

The crash is triggered by the combination of:
1. An immediate assertion statement (`assert (q == 1'b0)`)
2. An `else` clause with `$error()` containing a string literal message
3. Using arcilator as the backend

## Root Cause Analysis

### Issue Description

The arcilator pipeline processes SystemVerilog code through the following stages:
1. **Import**: `circt-verilog --ir-hw` converts SV to HW/Sim IR
2. **Preprocessing**: `StripSV` pass removes SV-specific constructs
3. **Arc Conversion**: Converts to Arc dialect for simulation
4. **LLVM Lowering**: `LowerArcToLLVM` converts Arc to LLVM IR

The problem occurs in the LLVM lowering stage. The `LowerArcToLLVM.cpp` pass marks certain `sim::Format*Op` operations as **legal** (not to be converted):

```cpp
// lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp
// Mark sim::Format*Op as legal. These are not converted to LLVM, but the
// lowering of sim::PrintFormattedOp walks them to build up its format string.
// They are all marked Pure so are removed after the conversion.
target.addLegalOp<sim::FormatLiteralOp, sim::FormatDecOp, sim::FormatHexOp,
                  sim::FormatBinOp, sim::FormatOctOp, sim::FormatCharOp,
                  sim::FormatStringConcatOp>();
```

**The assumption is that**:
1. `sim::Format*Op` operations are only used as inputs to `sim::PrintFormattedOp`
2. The lowering of `sim::PrintFormattedOp` will consume these format ops
3. Since they are marked `Pure`, DCE will remove them after lowering

**However**, when an immediate assertion with `$error()` is used:
1. The assertion creates a `sim.fmt.literal` for the error message
2. The `sim.fmt.literal` operation survives to the LLVM lowering phase
3. Without a corresponding `sim::PrintFormattedOp` that consumes it, the operation remains
4. Since it's marked "legal" but not actually converted or removed, it causes a legalization failure

### Missing Support

The arcilator pipeline lacks support for lowering immediate assertions (`assert ... else $error(...)`) to simulation code. The `sim.fmt.literal` operation produced for the `$error()` message has no consumer in the arcilator flow.

### Evidence

1. **Error message**: `failed to legalize operation 'sim.fmt.literal'`
2. **IR Context**: `%12 = "sim.fmt.literal"() <{literal = "Error: Assertion failed: q != 0"}>`
3. **Code comment in LowerArcToLLVM.cpp**: States Format*Op are expected to be consumed by `PrintFormattedOp` lowering
4. **StripSV.cpp**: Only strips `sv::AlwaysOp` but doesn't handle `sim` dialect assertion-related ops

## Root Cause Category

**Missing Implementation**: The arcilator pipeline does not have complete support for immediate assertions with action blocks (specifically, assertions with `else $error(...)` clauses). The `sim.fmt.literal` operation generated for the error message has no lowering path in the arcilator flow.

## Potential Fix Suggestions

### Option 1: Strip Sim Assertions in Arcilator Preprocessing
Extend the `StripSV` pass or create a new pass to strip or stub out assertion-related operations that cannot be lowered.

### Option 2: Implement Assertion Lowering in Arcilator
Implement proper lowering of `sim.fmt.literal` and related assertion ops to simulation-compatible code (e.g., `printf` calls).

### Option 3: Handle Unused Format Ops
Add DCE or cleanup to remove `sim::Format*Op` operations that have no users after the conversion process.

## Related Code Locations

| Location | Description |
|----------|-------------|
| `lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp` | Marks `sim::Format*Op` as legal |
| `lib/Dialect/Arc/Transforms/StripSV.cpp` | Strips SV-specific constructs for arcilator |
| `lib/Conversion/ImportVerilog/Statements.cpp` | Converts `$error()` to Moore IR |
| `lib/Tools/arcilator/pipelines.cpp` | Defines arcilator pass pipeline |

## Timestamp

2026-01-31T00:00:00Z
