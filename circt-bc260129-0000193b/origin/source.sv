module test_module(
  input logic clk,
  input signed [7:0] a,
  input real in_real,
  output real out_real
);

  logic cmp_result;

  always_comb begin
    cmp_result = (-a <= a) ? 1 : 0;
  end

  always @(posedge clk) begin
    out_real <= in_real;
  end

endmodule