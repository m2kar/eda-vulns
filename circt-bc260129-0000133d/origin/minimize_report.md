# CIRCT Bug Minimization Report

## Summary
- **Original file**: source.sv (15 lines)
- **Minimized file**: bug.sv (3 lines)
- **Reduction**: 80.0%

## Bug Description
The `sim.fmt.literal` operation generated from `$error` system task in an `always @(*)` assertion block fails to legalize in the ArcToLLVM pass.

## Key Construct
```systemverilog
always @(*) assert (condition) else $error("message");
```

## Reduction Steps
1. Removed output port (`result_out`)
2. Removed clock-driven `arr + 1` increment logic
3. Removed `idx` variable
4. Simplified `arr` variable to constant `0`
5. Shortened error message to empty string
6. Condensed to single-line format

## Original Test Case (15 lines)
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

## Minimized Test Case (3 lines)
```systemverilog
module m();
  always @(*) assert (0) else $error("");
endmodule
```

## Reproduction Command
```bash
circt-verilog --ir-hw bug.sv 2>&1 | arcilator 2>&1
```

## Error Output
```
<stdin>:3:10: error: failed to legalize operation 'sim.fmt.literal'
    %0 = sim.fmt.literal "Error: "
         ^
<stdin>:3:10: note: see current operation: %0 = "sim.fmt.literal"() <{literal = "Error: "}> : () -> !sim.fstring
```
