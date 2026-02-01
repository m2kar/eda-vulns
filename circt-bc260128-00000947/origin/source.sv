module MixPorts(
  input logic clk,
  input logic a,
  output logic b,
  inout wire c,
  output logic result
);

  logic intermediate_result;

  assign b = a;
  assign c = a ? 1'bz : 1'b0;

  always_comb begin
    intermediate_result = a ^ b;
  end

  always_ff @(posedge clk) begin
    result <= intermediate_result;
  end

endmodule