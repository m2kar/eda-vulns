# Root Cause Analysis Report

## Crash Summary

| Field | Value |
|-------|-------|
| **Testcase ID** | 260129-000015f7 |
| **Crash Type** | Assertion Failure |
| **Dialect** | LLHD |
| **Pass** | Mem2Reg |
| **Tool** | circt-verilog |

## Error Message

```
<unknown>:0: error: integer bitwidth is limited to 16777215 bits
Assertion `succeeded(ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

## Crash Location

The crash occurs in the LLHD Mem2Reg transformation pass:

```
#12 mlir::IntegerType::get(mlir::MLIRContext*, unsigned int, mlir::IntegerType::SignednessSemantics)
#13 (anonymous namespace)::Promoter::insertBlockArgs((anonymous namespace)::BlockEntry*)
    at lib/Dialect/LLHD/Transforms/Mem2Reg.cpp:1742:35
#14 (anonymous namespace)::Promoter::insertBlockArgs()
    at lib/Dialect/LLHD/Transforms/Mem2Reg.cpp:1654:28
#15 (anonymous namespace)::Promoter::promote()
    at lib/Dialect/LLHD/Transforms/Mem2Reg.cpp:764:3
#16 (anonymous namespace)::Mem2RegPass::runOnOperation()
    at lib/Dialect/LLHD/Transforms/Mem2Reg.cpp:1844:34
```

## Problematic Test Case

```systemverilog
module test_module(input logic clk, input logic rst);

  // Class declaration with properties
  class my_class;
    logic [7:0] data;
    function void set_data(logic [7:0] val);
      data = val;
    endfunction
  endclass

  // Signal connecting combinational and sequential logic
  logic [7:0] computed_value;
  
  // Class object for sequential logic
  my_class mc;

  // Combinational logic block
  always @(*) begin
    computed_value = 8'hAA;  // Example combinational logic
  end

  // Sequential logic block with class instantiation
  always @(posedge clk) begin
    if (rst) begin
      mc = new();
      mc.set_data(8'h00);
    end else begin
      mc = new();
      mc.set_data(computed_value);
    end
  end

endmodule
```

## Root Cause Analysis

### Immediate Cause
The LLHD Mem2Reg pass attempts to create an MLIR IntegerType with a bitwidth exceeding the maximum allowed limit (16777215 bits). This occurs during the `insertBlockArgs` phase of the memory-to-register promotion optimization.

### Underlying Cause
The crash is triggered by SystemVerilog class constructs that are explicitly noted as "not yet supported":

1. **Unsupported Class Feature**: The warning message confirms: "Class builtin functions (needed for randomization, constraints, and covergroups) are not yet supported and will be dropped during lowering."

2. **Class Object Variable**: The variable `my_class mc;` declares a class-type variable that is used in sequential logic blocks.

3. **Type Mishandling**: When the Mem2Reg pass encounters this class-type variable during block argument insertion:
   - It attempts to determine the bitwidth of the type
   - The class type's size is incorrectly interpreted or computed
   - This results in an absurdly large bitwidth value (> 16777215 bits)
   - The `IntegerType::get()` call fails its verification invariants

### Technical Details

The crash path shows:
1. `Mem2RegPass::runOnOperation()` initiates the pass
2. `Promoter::promote()` is called to perform memory-to-register promotion
3. `Promoter::insertBlockArgs()` attempts to add block arguments for promoted values
4. When processing the class-type variable, it calls `IntegerType::get()` with an invalid bitwidth
5. MLIR's invariant verification fails, triggering the assertion

### Key Observations

1. **Missing Type Validation**: The Mem2Reg pass does not properly validate or handle class types before attempting to convert them to MLIR integer types.

2. **Graceful Degradation Missing**: Although classes are marked as unsupported and "dropped during lowering," the lowering process doesn't fully remove or properly handle all class-related constructs before the Mem2Reg pass runs.

3. **Type System Gap**: There's a mismatch between what types the Mem2Reg pass expects to handle and what types can appear in the IR after partial class lowering.

## Recommended Fix

1. **Input Validation**: The Mem2Reg pass should check for unsupported types (like class types) and skip or gracefully handle them instead of attempting promotion.

2. **Type Guard**: Before calling `IntegerType::get()`, validate that the computed bitwidth is within MLIR's limits.

3. **Complete Lowering**: Ensure class-related constructs are fully lowered or removed before passes that cannot handle them.

4. **Error Emission**: Instead of crashing on assertion, emit a proper diagnostic error when encountering unsupported types during optimization passes.

## Classification

- **Bug Category**: Type Handling / Unsupported Feature Crash
- **Severity**: Medium (crashes on unsupported but valid SystemVerilog)
- **Reproducibility**: Deterministic
- **Workaround**: Avoid using SystemVerilog classes in designs processed by circt-verilog
