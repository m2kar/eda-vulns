# Root Cause Analysis: sim.fmt.literal Legalization Failure

## Error Summary

- **Error Type**: Legalization failure
- **Failed Operation**: `sim.fmt.literal`
- **Error Message**: `failed to legalize operation 'sim.fmt.literal'`
- **Tool Pipeline**: `circt-verilog --ir-hw` -> `arcilator` -> `opt -O0` -> `llc -O0`

## Source Code Analysis

### Problematic SystemVerilog Code

```systemverilog
module array_processor(input logic clk, output logic [7:0] result_out);
  logic [15:0] arr;
  int idx;
  
  always @(posedge clk) begin
    arr <= arr + 1;
  end
  
  always @(*) begin
    idx = 0;
    assert (arr[idx] == 1'b1) else $error("Assertion failed: arr[%0d] != 1", idx);
  end
  
  assign result_out = arr[7:0];
endmodule
```

### Key Observations

1. **Combinational Assertion**: The `assert` statement with `$error` system task is inside a combinational `always @(*)` block
2. **Dynamic Index**: The assertion uses a dynamic index `idx` which is set at runtime
3. **Format String**: The `$error` call includes a format string with `%0d` format specifier and variable argument `idx`

## Root Cause Hypothesis

### Primary Root Cause: Missing sim.fmt.literal Pattern in ArcToLLVM Conversion

The `sim.fmt.literal` operation is part of CIRCT's Sim dialect, which provides high-level simulation constructs. When the `$error` system task is translated through the CIRCT pipeline:

1. **ImportVerilog Phase**: `$error("Assertion failed: arr[%0d] != 1", idx)` is converted to sim dialect operations:
   - `sim.fmt.literal "Error: Assertion failed: arr[0] != 1"` (constant folded message)

2. **Arcilator Phase**: The arcilator tool processes the IR for simulation, but the `sim.fmt.literal` operation lacks a proper lowering pattern in the ArcToLLVM conversion pass.

### Technical Details

From examining the CIRCT source:

1. **SimToSV Pass** (`lib/Conversion/SimToSV/SimToSV.cpp`): Provides conversion patterns for:
   - `PlusArgsTestOp`, `PlusArgsValueOp`
   - `ClockedTerminateOp`, `ClockedPauseOp`
   - `TerminateOp`, `PauseOp`
   - `DPICallOp`
   - **But NOT `FormatLiteralOp` or other formatting operations**

2. **ArcToLLVM Pass** (`lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp`): Handles:
   - `SimPrintFormattedProcOp` via `foldFormatString()` which supports:
     - `FormatCharOp`, `FormatDecOp`, `FormatHexOp`, `FormatOctOp`
     - `FormatLiteralOp`, `FormatStringConcatOp`
   - However, `sim.print` (PrintFormattedOp) and non-procedural contexts may not be fully handled

3. **The Gap**: When `sim.fmt.literal` appears in a non-procedural context (combinational always block), the dialectConversion framework marks `SimDialect` as illegal but no pattern handles `FormatLiteralOp` in isolation outside of a `PrintFormattedProcOp` context.

### Why This Happens

1. The assertion with `$error` in combinational logic generates `sim.fmt.literal` outside the expected procedural context
2. `SimToSV` does not provide patterns for format string operations (they are meant to be used inside print operations)
3. `ArcToLLVM` expects format operations to be consumed by `SimPrintFormattedProcOp`, not to appear standalone
4. When `sim.fmt.literal` is orphaned (not part of a print hierarchy), no conversion pattern matches, causing legalization failure

## Suggested Fix Direction

1. **Option A**: Add a pattern to lower standalone `sim.fmt.literal` to a no-op or constant in ArcToLLVM when not used by a print operation

2. **Option B**: In ImportVerilog, avoid generating `sim.fmt.literal` for assertions that are constant-folded to pure string literals, or generate `sim.print`/`sim.proc.print` wrapper

3. **Option C**: Add DCE (dead code elimination) before ArcToLLVM to remove unused format string operations

4. **Option D**: Update the conversion target in ArcToLLVM to mark `sim.fmt.literal` as legal when orphaned, allowing it to be eliminated later

## Related CIRCT Components

- `lib/Conversion/ImportVerilog/Statements.cpp` - Assertion translation
- `lib/Conversion/SimToSV/SimToSV.cpp` - Sim to SV lowering
- `lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp` - Arc to LLVM lowering
- `include/circt/Dialect/Sim/SimOps.td` - Sim dialect operation definitions
