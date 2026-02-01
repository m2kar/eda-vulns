# Minimization Report

## Original Test Case

Original file: `source.sv` (31 lines, 772 bytes)

```systemverilog
module MixedPorts(
  input  logic a,
  input  logic clk,
  output logic b,
  inout  logic c
);

  typedef enum logic {STATE_A, STATE_B} state_t;
  state_t curr_state, next_state;

  // Sequential state update using the clock
  always_ff @(posedge clk) begin
    curr_state <= next_state;
  end

  // Combinational logic to set next_state based on input and current state
  always_comb begin
    case (curr_state)
      STATE_A: next_state = a ? STATE_B : STATE_A;
      STATE_B: next_state = STATE_A;
      default: next_state = STATE_A;
    endcase
  end

  // Assignment of state-dependent value to output
  assign b = (curr_state == STATE_B);

  // Inout handling: high-Z when STATE_A, drive 0 when STATE_B
  assign c = (curr_state == STATE_A) ? 1'bz : 1'b0;

endmodule
```

## Minimized Test Case

Minimized file: `bug.sv` (2 lines, 31 bytes)

```systemverilog
module Bug(inout logic c);
endmodule
```

## Reduction Statistics

- **Original size**: 772 bytes, 31 lines
- **Minimized size**: 31 bytes, 2 lines
- **Reduction**: 96.0% (741 bytes removed)
- **Line reduction**: 93.5% (29 lines removed)

## Key Findings

The crash is triggered solely by the presence of an `inout` port. All other constructs (input/output ports, state machines, combinational logic, sequential logic) are irrelevant to the bug.

### Removed Constructs (not needed for crash)
- Input ports
- Output ports  
- Clock/sequential logic (always_ff)
- Combinational logic (always_comb)
- typedef/enum
- State variables
- Conditional assignments

### Preserved Constructs (essential for crash)
- Module declaration
- `inout` port declaration

## Verification

Both test cases produce the same crash:

```
error: state type must have a known bit width; got '!llhd.ref<i1>'
```

Stack trace location: `LowerState.cpp:219` in `ModuleLowering::run()`
