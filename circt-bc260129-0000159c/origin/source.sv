module MixedPorts(
  input  logic clk,
  input  logic a,
  output logic b,
  inout  logic c
);

  logic [3:0] temp_reg;

  always_ff @(posedge clk) begin
    for (int i = 0; i < 4; i++) begin
      temp_reg[i] = a & i[0];
    end
  end

  assign b = temp_reg[0];

  assign c = (a) ? temp_reg[0] : 1'bz;

endmodule