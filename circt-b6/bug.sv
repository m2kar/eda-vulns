module MixedPorts(
  inout wire c,
  input logic clk
);
  logic temp_reg;

  always_ff @(posedge clk) begin
    temp_reg <= c;
  end
endmodule
