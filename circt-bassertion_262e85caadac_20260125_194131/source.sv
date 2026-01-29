module example_module(input logic clock, output logic q_out);
  // Internal wire declarations
  wire _00_, _01_, _02_;
  
  // Data input signal for the flip-flop
  logic d;
  
  // Assignments to drive internal wires
  assign _00_ = clock;
  assign _01_ = ~clock;
  assign _02_ = _00_ & _01_;
  
  // Always block with negedge clock sensitivity
  always @(negedge clock) begin
    q_out <= d;
  end
  
  // Assignment to connect internal wire to output
  assign q_out = _02_ ? 1'b0 : q_out;
endmodule