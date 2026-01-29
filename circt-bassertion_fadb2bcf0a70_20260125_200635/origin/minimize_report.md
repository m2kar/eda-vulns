# Minimize Report

## Summary
- **Original Lines**: 27
- **Minimized Lines**: 4
- **Reduction**: 85.2%
- **Crash Preserved**: ✅ Yes

## Minimization Strategy

Based on `analysis.json`:
- **Minimal Trigger**: `output string str`
- **Key Construct**: string type used as module output port

### Removed Elements
1. Input ports: `clock`, `reset`, `enable` (not needed for crash)
2. Internal signals: `d`, `q` (not needed)
3. Always block with register logic (not needed)
4. Assign statement (not needed)
5. always_comb block with `$sformatf` (not needed)
6. Comments (not needed)

### Preserved Elements
- Module declaration with `output string str` port

## Original Test Case (27 lines)
```systemverilog
module example(
  input  logic clock,
  input  logic reset,
  input  logic enable,
  output string str
);

  logic [31:0] d;
  logic [31:0] q;

  // Register with asynchronous reset and enable condition
  always @(posedge clock, posedge reset) begin
    if (reset)
      q <= 32'd42;
    else if (enable)
      q <= d;
  end

  // Connect register output to data input (increment feedback)
  assign d = q + 1;

  // String assignment using register value
  always_comb begin
    str = $sformatf("Value: %0d", q);
  end

endmodule
```

## Minimized Test Case (4 lines)
```systemverilog
module example(
  output string str
);
endmodule
```

## Verification
The minimized test case triggers the same assertion failure:
```
Assertion `detail::isPresent(Val) && "dyn_cast on a non-existent value"' failed.
```

At the same location:
- File: `MooreToCore.cpp:259`
- Function: `getModulePortInfo()`
- Pass: `MooreToCorePass`

## Conclusion
The crash is triggered solely by declaring a `string` type output port. All other code in the original test case was irrelevant to the bug.
