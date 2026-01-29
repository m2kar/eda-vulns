module MixedPorts(
  input  logic clk,
  input  logic rst,
  input  logic a,
  output logic b,
  inout  logic c
);

  logic b_reg;

  always_ff @(posedge clk) begin
    if (!rst) begin
      b_reg <= 1'b0;
    end else begin
      b_reg <= a;
    end
  end

  assign b = b_reg;
  assign c = (!rst) ? 1'bz : a;

endmodule