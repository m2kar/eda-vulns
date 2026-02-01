module test_module(
  input real in_real,
  input logic clk,
  output real out_real,
  output logic cmp_result
);

  // Continuous assignment comparing real input against threshold
  assign cmp_result = (in_real > 2.5);

  // Sequential assignment updating real output on clock edge
  always_ff @(posedge clk) begin
    out_real <= in_real * 0.9;
  end

endmodule