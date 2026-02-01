# CIRCT Crash Root Cause Analysis

## Summary

**Crash ID**: bc260129-000014ab  
**Dialect**: LLHD  
**Crash Type**: Assertion failure  
**Location**: `lib/Dialect/LLHD/Transforms/Mem2Reg.cpp:1742`

## Error Context

```
<unknown>:0: error: integer bitwidth is limited to 16777215 bits
Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

The crash occurs during the LLHD Mem2Reg pass when attempting to create an `IntegerType` with a bitwidth exceeding MLIR's maximum supported value of 16,777,215 bits (2^24 - 1).

## Testcase Analysis

The testcase (`source.sv`) contains:

```systemverilog
module top(input logic clk, input logic rst, input logic cond);
  
  class my_class;
    function new();
    endfunction
  endclass
  
  my_class mc_handle;
  
  always @(posedge clk) begin
    if (rst) begin
      my_class mc;
      mc = new();
      mc_handle = mc;
    end
  end
  
  always @(posedge clk) begin
    if (cond) begin
      // empty block
    end
  end
  
endmodule
```

### Key Problematic Constructs

1. **SystemVerilog Class Definition**: `my_class` is defined as a class type
2. **Class Handle Variable**: `my_class mc_handle` declares a class handle (reference type)
3. **Always Block with Class Operations**: Local class variable `mc` is created and assigned inside an always block
4. **Class Instantiation**: `mc = new()` creates a class instance

## Root Cause Hypothesis

### Primary Issue: Unsupported Class Type in Mem2Reg Transformation

The LLHD Mem2Reg pass (`lib/Dialect/LLHD/Transforms/Mem2Reg.cpp`) is designed to promote memory slots (signals) to SSA values. In the function `Promoter::insertBlockArgs()` at line ~1748, the code attempts to compute a "flat" integer type for block arguments:

```cpp
case Which::Value:
  if (def) {
    args.push_back(def->getValueOrPlaceholder());
  } else {
    auto type = getStoredType(slot);
    auto flatType = builder.getIntegerType(hw::getBitWidth(type));  // Line ~1753
    Value value =
        hw::ConstantOp::create(builder, getLoc(slot), flatType, 0);
    // ...
  }
```

The function `hw::getBitWidth(type)` is called to determine the bit width of the stored type. For class types (which are reference/handle types in SystemVerilog), this function likely:

1. Returns `-1` (indicating unknown/unsupported type), which then gets cast to unsigned and becomes a huge positive number
2. Or returns an unexpectedly large value due to incorrect handling of class types

When this incorrect/huge value is passed to `builder.getIntegerType()`, it triggers MLIR's `IntegerType::get()` verification which rejects widths > 16,777,215.

### Contributing Factors

1. **Partial Class Support**: The warning message indicates class builtin functions are not yet supported:
   ```
   remark: Class builtin functions (needed for randomization, constraints, and covergroups) are not yet supported and will be dropped during lowering.
   ```
   This suggests class types are partially supported but the lowering/transformation pipeline doesn't fully handle them.

2. **Missing Type Validation**: The Mem2Reg pass doesn't validate that the slot's stored type can be legally converted to an integer type before calling `hw::getBitWidth()`.

3. **Implicit Signal Creation**: The class handle `mc_handle` and local variable `mc` are likely converted to LLHD signals, which the Mem2Reg pass then tries to promote.

## Call Stack Analysis

```
#12 mlir::IntegerType::get                   // MLIR tries to create IntegerType
#13 Promoter::insertBlockArgs(BlockEntry*)   // Line 1742 - tries to create block arg
#14 Promoter::insertBlockArgs()              // Iterates over blocks
#15 Promoter::promote()                      // Main promotion entry
#16 Mem2RegPass::runOnOperation()            // Pass invocation
```

The crash happens at the lowest level when MLIR attempts to create an integer type, but the actual bug originates from the Mem2Reg pass not properly handling class/handle types.

## Trigger Pattern

The bug is triggered when:
1. A SystemVerilog module contains a class definition
2. A class handle variable is declared
3. The handle is assigned inside a procedural block (always block)
4. The LLHD Mem2Reg transformation attempts to promote the signal associated with the class handle

## Recommended Fix

1. **Add Type Validation**: Before calling `hw::getBitWidth()`, check if the type is a supported hardware type that can be represented as an integer.

2. **Skip Unsupported Slots**: Modify `findPromotableSlots()` to exclude slots with class/handle types from promotion.

3. **Proper Error Handling**: If an unsupported type is encountered, emit a diagnostic and skip the transformation rather than proceeding with an invalid bitwidth.

## Classification

- **Bug Type**: Assertion failure / missing input validation
- **Severity**: High (causes crash)
- **Reproducibility**: Deterministic
- **Root Cause Category**: Incomplete feature support - class types not properly handled in LLHD transformations
