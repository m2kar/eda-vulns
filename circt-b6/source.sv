module MixedPorts(
  input logic a,
  output logic b,
  inout wire c,
  input logic clk
);

  logic [3:0] temp_reg;

  // Sized decimal constant assignment to output
  assign b = 4'd2;

  // Always block using inout port and clock
  always_ff @(posedge clk) begin
    temp_reg <= c;
  end

  // Bidirectional assignment for inout port
  assign c = a ? temp_reg : 4'bz;

endmodule