# CIRCT Bug Report: sim.fmt.literal Legalization Failure in ArcToLLVM Conversion

## Title
**[Arcilator] ArcToLLVM conversion fails to legalize `sim.fmt.literal` operations created from SystemVerilog assertions**

## Body

### Description

When compiling SystemVerilog code containing immediate assertions with format strings through the circt-verilog → arcilator pipeline, the ArcToLLVM conversion pass fails with a legalization error for `sim.fmt.literal` operations.

#### Test Case
```systemverilog
module test_module(
  input logic clk,
  output logic [7:0] result
);

  // Array declaration and index variable
  logic [7:0] arr [0:7];
  logic [2:0] idx = 3'b0;

  // Function that increments a value
  function automatic [7:0] add_one(input [7:0] val);
    add_one = val + 8'h1;
  endfunction

  // Combinational logic with function call
  always_comb begin
    result = add_one(8'h5);
  end

  // Immediate assertion with array access and format string
  always @(*) begin
    assert (arr[idx] == 1'b1) else $error("Assertion failed: arr[%0d] != 1", idx);
  end

endmodule
```

#### Error Message
```
<stdin>:6:10: error: failed to legalize operation 'sim.fmt.literal'
    %0 = sim.fmt.literal "Error: Assertion failed: arr["
         ^
<stdin>:6:10: note: see current operation: %56 = "sim.fmt.literal"() <{literal = "Error: Assertion failed: arr["}> : () -> !sim.fstring
```

#### Reproduction Command
```bash
export PATH=/opt/llvm-22/bin:$PATH && \
circt-verilog --ir-hw test_module.sv | \
arcilator | \
opt -O0
```

### Expected Behavior
The assertion with format string should compile successfully and generate object code.

### Actual Behavior
ArcToLLVM conversion fails with "failed to legalize operation 'sim.fmt.literal'" error.

## Root Cause

The bug is in the **ArcToLLVM conversion pass** handling of format string operations that are created from SystemVerilog assertions.

### Analysis

1. **Format String Operation Generation**: When `circt-verilog` processes the `$error()` system task with a format string containing `%0d` format specifier, it generates:
   - `sim.fmt.literal "Error: Assertion failed: arr["`
   - `sim.fmt.literal "] != 1"`
   - `sim.fmt.dec %idx specifierWidth 0`
   - `sim.fmt.concat (%0, %9, %1)`
   - `sim.proc.print %10`

2. **LLHD Combinational Region**: The format operations appear inside an `llhd.combinational` region in the generated MLIR.

3. **Arc Conversion Pipeline**: The `llhd.combinational` region is converted to an `arc.execute` operation by the `ConvertToArcs` pass, preserving the format operations inside.

4. **ArcToLLVM Conversion Target**: In `LowerArcToLLVM.cpp` (lines 1241-1243), format operations are marked as LEGAL:
   ```cpp
   target.addLegalOp<sim::FormatLiteralOp, sim::FormatDecOp, sim::FormatHexOp,
                     sim::FormatBinOp, sim::FormatOctOp, sim::FormatCharOp,
                     sim::FormatStringConcatOp>();
   ```
   The comment states: "These are not converted to LLVM, but lowering of sim::PrintFormattedOp walks them to build up its format string. They are all marked Pure so are removed after conversion."

5. **Conversion Pattern**: `SimPrintFormattedProcOpLowering` (lines 874-902) is registered and should:
   - Call `foldFormatString()` to process the format tree
   - Replace `sim.proc.print` with a runtime call
   - Format operations should become dead and be eliminated

6. **The Problem**: The format operations survive through the conversion and are still present when the LLVM dialect conversion framework tries to verify all operations are legal. This causes:
   - Legalization failure for `sim.fmt.literal` operations
   - Error: "failed to legalize operation 'sim.fmt.literal'"

### Potential Issue

The `arc.execute` lowering pattern inlines the body region (line 937: `rewriter.inlineRegionBefore(op.getBody(), blockAfter)`), which may cause scoping issues where format operations are moved outside their expected context. Additionally, the type conversion of `!sim.fstring` to `LLVM::LLVMPointerType` (line 1262-1264) creates pointers without pointee types, which may cause validation issues.

## Files Affected

- **Conversion**: `circt/lib/Conversion/ArcToLLVM/LowerArcToLLVM.cpp`
  - Lines 874-902: `SimPrintFormattedProcOpLowering` pattern
  - Lines 1241-1243: Conversion target marking format ops as legal
  - Lines 1262-1264: Type conversion for FormatStringType

- **Operation Definition**: `circt/include/circt/Dialect/Sim/SimOps.td`
  - Lines 155-171: `FormatLiteralOp` definition

## Steps to Reproduce

1. Create SystemVerilog test file with immediate assertion using `$error()` with format string
2. Run: `circt-verilog --ir-hw test.sv | arcilator`
3. Observe the legalization failure

## Minimal Test Case

```systemverilog
module test_minimal(
  input logic clk
);

initial begin
  assert (1'b0 == 1'b1) else $error("Assertion failed");
end

endmodule
```

## Additional Notes

- The issue appears to be related to the nascent support for sim.proc.print and sim.fmt.* operations added in commit f40496973 (January 19, 2026).
- Git history shows recent work on Arc simulation backend with format string support.
- This is a blocker for using SystemVerilog assertions with formatted error messages in simulation workflows.
- The error occurs during ArcToLLVM conversion stage, NOT in LLVM opt phase.
