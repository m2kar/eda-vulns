module test_module(
  input logic clk,
  input logic a,
  output logic b
);

  always_ff @(posedge clk) begin
    b <= ~a;
  end

endmodule