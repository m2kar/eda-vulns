module MixPorts(
  input logic a,
  output logic b,
  inout wire c,
  input logic clk,
  output logic out0
);

  logic [3:0] idx;

  always_ff @(posedge clk) begin
    idx <= idx + 1;
  end

  assign out0 = idx[0];
  assign b = a;
  assign c = a ? 1'bz : 1'b0;

endmodule