module test_module(
  input logic clk,
  output logic r1_out
);

  logic r1;

  always_ff @(posedge clk) begin
    r1 <= ~r1;
  end

  assign r1_out = r1;

endmodule