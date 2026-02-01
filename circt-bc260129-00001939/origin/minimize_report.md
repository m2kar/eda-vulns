# Test Case Minimization Report

## Summary

- **Original file**: source.sv (18 lines)
- **Minimized file**: bug.sv (3 lines)
- **Reduction**: 83.3% (15 lines removed)

## Original Test Case

```systemverilog
module test_module(
    input real in_real,
    input logic clk,
    output real out_real,
    output logic cmp_result
);

    wire signed [1:0] a = 2'b10;
    
    always_comb begin
        cmp_result = (-a <= a) ? 1 : 0;
    end
    
    always @(posedge clk) begin
        out_real <= in_real;
    end

endmodule
```

## Minimized Test Case

```systemverilog
module m(input c, output real o);
always @(posedge c) o <= 0;
endmodule
```

## Reduction Process

### Phase 1: Initial Analysis
Based on the root cause analysis, the trigger pattern was identified as:
- Signed wire with unary negation in comparison expression
- Clocked assignment to real output port

### Phase 2: Iterative Reduction

| Step | Change | Result | Lines |
|------|--------|--------|-------|
| 1 | Removed all ports except clk, kept signed wire, always_comb with ternary | No crash | 5 |
| 2 | Added real input/output back | Crash | 13 |
| 3 | Simplified always_comb to single-line | Crash | 9 |
| 4 | Removed cmp_result output | Crash | 7 |
| 5 | Removed in_real input | Crash | 6 |
| 6 | Removed cmp_result wire, inline comparison | Crash | 5 |
| 7 | Removed comparison, just negation | Crash | 4 |
| 8 | Removed negation, just variable | Crash | 4 |
| 9 | Replaced variable with constant | Crash | 4 |
| 10 | Simplified to minimal | Crash | 3 |

### Phase 3: Key Discoveries

The bug is **simpler than initially analyzed**. The actual minimal trigger is:

1. **Real output port** (`output real o`)
2. **Clocked always block** (`always @(posedge c)`)
3. **Assignment to real output** (`o <= 0`)

The signed wire, unary negation, and comparison operators were **not essential** to trigger the crash. The core issue is that clocked assignments to `real` type outputs cause the Mem2Reg pass to fail when trying to create an IntegerType for the floating-point register.

## Crash Signature Verification

Both original and minimized test cases produce the same assertion failure:
```
<unknown>:0: error: integer bitwidth is limited to 16777215 bits
Assertion `succeeded(ConcreteT::verifyInvariants(...))' failed.
```

Stack trace confirms same crash location:
- `lib/Dialect/LLHD/Transforms/Mem2Reg.cpp:1742` in `Promoter::insertBlockArgs`

## Conclusion

The minimization discovered that the bug is more fundamental than initially analyzed:
- **Reported trigger**: Signed wire with unary negation in comparison
- **Actual trigger**: Any clocked assignment to a `real` type output

This suggests the Mem2Reg pass does not properly handle floating-point (`real`) types when processing sequential logic, attempting to create an IntegerType for what should be a floating-point register.
