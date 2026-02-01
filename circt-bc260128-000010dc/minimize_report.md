# Minimization Report

## Summary

| Metric | Value |
|--------|-------|
| Original Size | 749 bytes |
| Minimized Size | 545 bytes |
| Reduction | 204 bytes |
| **Reduction Percent** | **27.2%** |
| Original Lines | 36 |
| Minimized Lines | 24 |

## Minimization Process

### Key Constructs Identified

Based on root cause analysis (`InferStateProperties.cpp:211` type mismatch):
1. **Two array declarations** - Required for enable pattern detection to trigger
2. **for loop** - Required for pattern matching that leads to crash
3. **Conditional assignment** - Triggers mux-based enable detection
4. **Two outputs** - Required to use both arrays

### Iterative Reduction Steps

| Step | Change | Result |
|------|--------|--------|
| 1 | Remove DATA_WIDTH parameter | ✅ Crash |
| 2 | Remove data variable | ✅ Crash |
| 3 | Remove idx state variable | ✅ Crash |
| 4 | Simplify array size to [0:1] | ✅ Crash |
| 5 | Reduce for loop to single iteration | ✅ Crash |
| 6 | Remove packed_arr loop assignment | ✅ Crash |
| 7 | Remove one array entirely | ❌ No Crash |
| 8 | Remove for loop | ❌ No Crash |
| 9 | Remove packed_arr[0] assignment | ❌ No Crash |

### Essential Components

The minimal testcase requires ALL of the following:
1. **Two unpacked arrays**: `packed_arr` and `unpacked_arr`
2. **for loop**: Even single iteration is sufficient
3. **Conditional assignment**: `if-else` on one array element
4. **Assignment to second array**: At least one assignment to `packed_arr`
5. **Output usage**: Both arrays must be read

## Root Cause Connection

The crash occurs because:
1. `InferStatePropertiesPass` detects enable patterns in the `for` loop combined with conditional assignment
2. `applyEnableTransformation` attempts to create `hw::ConstantOp` with array type
3. `hw::ConstantOp::create` expects `IntegerType` but receives `ArrayType`
4. Assertion failure in `llvm::cast<mlir::IntegerType>()`

## Minimized Testcase

```systemverilog
module test_module(
    input logic clk,
    input logic [7:0] data_in,
    output logic [7:0] out_unpacked,
    output logic [7:0] out_packed
);
  logic [7:0] packed_arr [0:1];
  logic [7:0] unpacked_arr [0:1];

  always_ff @(posedge clk) begin
    if (data_in == 0)
      unpacked_arr[0] <= 8'hFF;
    else
      unpacked_arr[0] <= data_in;
    
    for (int i = 1; i < 2; i++)
      unpacked_arr[i] <= data_in;
    
    packed_arr[0] <= data_in;
  end

  assign out_unpacked = unpacked_arr[0];
  assign out_packed = packed_arr[0];

endmodule
```

## Reproduction

```bash
/edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/circt-verilog --ir-hw bug.sv | /edazz/FeatureFuzz-SV/target/circt-1.139.0/bin/arcilator
```

## Notes

- Bug is specific to CIRCT version 1.139.0 (firtool-1.139.0)
- Newer CIRCT versions (690366b6c) show different behavior (type mismatch error instead of crash)
- The for loop is critical - static array assignments do not trigger the bug
