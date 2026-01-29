module MixedPorts(
  input  logic clk,
  input  logic A,
  output logic O,
  inout  logic C
);

  // Blocking assignment inside clocked always block
  always @(posedge clk) begin
    O = A;
  end

  // Signal assignment to drive the inout port
  assign C = A;

endmodule