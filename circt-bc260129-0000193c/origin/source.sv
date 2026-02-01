module test_module(
  input logic clk,
  input real in_real,
  output real out_real,
  output logic cmp_result
);

  always_ff @(posedge clk) begin
    out_real <= in_real * 2.0;
    cmp_result <= (in_real > 5.0);
  end

endmodule