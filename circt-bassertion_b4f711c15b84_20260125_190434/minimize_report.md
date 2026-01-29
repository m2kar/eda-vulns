# Minimize Report

## Summary

| Metric | Value |
|--------|-------|
| Original Lines | 20 |
| Minimized Lines | 2 |
| Reduction | 90% |
| Crash Reproducible | Yes |

## Original Test Case (source.sv)

```systemverilog
module MixedPorts(
  input logic clk,
  input logic a,
  output logic b,
  inout wire c
);

  logic [7:0] count;

  always_ff @(posedge clk) begin
    if (a) begin
      count <= 8'd0;
    end else begin
      count <= count + 8'd1;
    end
  end

  assign b = count[0];

endmodule
```

## Minimized Test Case (bug.sv)

```systemverilog
module MixedPorts(inout wire c);
endmodule
```

## Minimization Steps

1. **Remove empty lines and compress formatting** - Crash persists ✓
2. **Remove `always_ff` block and `count` variable** - Crash persists ✓
3. **Remove `clk`, `a`, `b` ports and `assign` statement** - Crash persists ✓
4. **Final form: single `inout wire` port** - Crash persists ✓

## Key Constructs Preserved

Based on `analysis.json`, the following key constructs were preserved:

- **`inout wire`**: This is the trigger for the crash. The `inout` port creates an `llhd::RefType` which is not supported by `StateType::get()` in arcilator's LowerStatePass.

## Constructs Removed

- `input logic clk` - not relevant to crash
- `input logic a` - not relevant to crash  
- `output logic b` - not relevant to crash
- `logic [7:0] count` - internal state variable, not relevant
- `always_ff` block - sequential logic, not relevant
- `assign b = count[0]` - combinational logic, not relevant

## Verification

The minimized test case reproduces the exact same crash:

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
Assertion `succeeded( ConcreteT::verifyInvariants(...))' failed.
```

Stack trace confirms crash at:
- `LowerState.cpp:219` in `ModuleLowering::run()`
- `StateType::get()` fails for `llhd::RefType`

## Conclusion

The crash is triggered by the minimal construct of a module with an `inout wire` port. The arcilator tool's LowerStatePass attempts to create a `StateType` for the port, but `llhd::RefType` (used for inout ports) is not a supported type for state storage.
