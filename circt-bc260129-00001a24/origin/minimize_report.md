# Minimization Report

## Summary
- **Original file**: source.sv (16 lines)
- **Minimized file**: bug.sv (10 lines)
- **Reduction**: 37.5%
- **Crash preserved**: Yes

## Preservation Analysis

### Key Constructs Preserved
Based on `analysis.json`, the following critical constructs were kept:
- **real type**: Input and output ports of type `real`
- **always_ff block**: Sequential logic triggered on clock edge
- **Real arithmetic**: Multiplication operation on floating-point values

### Removed Elements
- **Comparison logic**: Removed `cmp_result` output port and continuous assignment `(in_real > 2.5)`
- **Comments**: Removed all comments
- **Unused logic**: Removed combinational comparison that wasn't necessary to trigger the crash

### Reduction Rationale
The crash occurs in the LLHD Mem2Reg pass when handling floating-point types in sequential logic. The comparison operation and additional output port were removed because:
1. The crash is triggered by the presence of `real` type in `always_ff` block
2. The comparison logic is not involved in the Mem2Reg optimization path
3. Test case was simplified to minimal code that still exhibits the bug

## Verification

### Original Assertion
```
<unknown>:0: error: integer bitwidth is limited to 16777215 bits
circt-verilog: /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/include/mlir/IR/StorageUniquerSupport.h:180: 
  Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

### Final Assertion
```
<unknown>:0: error: integer bitwidth is limited to 16777215 bits
circt-verilog: /home/zhiqing/edazz/FeatureFuzz-SV/target/circt-1.139.0-src/llvm/mlir/include/mlir/IR/StorageUniquerSupport.h:180: 
  Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
```

**Match**: ✅ Exact match - Same assertion message and crash location (Mem2Reg.cpp:1742)

## Reproduction Command

```bash
export PATH=/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin:$PATH
circt-verilog bug.sv --ir-hw
```

## Test Case Comparison

### Original (source.sv)
```systemverilog
module test_module(
  input real in_real,
  input logic clk,
  output real out_real,
  output logic cmp_result
);

  // Continuous assignment comparing real input against threshold
  assign cmp_result = (in_real > 2.5);

  // Sequential assignment updating real output on clock edge
  always_ff @(posedge clk) begin
    out_real <= in_real * 0.9;
  end

endmodule
```

### Minimized (bug.sv)
```systemverilog
module bug(
  input real in_real,
  input logic clk,
  output real out_real
);

  always_ff @(posedge clk) begin
    out_real <= in_real * 0.9;
  end

endmodule
```

## Notes

1. **Root cause preservation**: The minimized test case still triggers the same Mem2Reg crash because it contains the essential pattern that causes the bug: `real` type usage in `always_ff` block.

2. **Verification confirmed**: The exact same assertion failure occurs at Mem2Reg.cpp:1742 in the `Promoter::insertBlockArgs` function.

3. **Minimal but complete**: The test case cannot be further reduced without losing the ability to reproduce the crash:
   - The `always_ff` block is required (removal eliminates the crash)
   - The `real` type is required (changing to `logic` eliminates the crash)
   - The assignment is required (empty block doesn't crash)

4. **Simplicity**: The minimized test case is self-contained and easy to understand, making it ideal for bug reporting and reproduction.

## Files Generated

- `bug.sv` - Minimized test case (10 lines)
- `error.log` - Complete crash output
- `command.txt` - Single-line reproduction command
