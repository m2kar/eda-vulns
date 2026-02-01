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