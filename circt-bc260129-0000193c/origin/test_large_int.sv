module test_module(
  input logic clk,
  output logic out
);

  logic [100000000:0] large_vec;  // Very large vector
  
  always_ff @(posedge clk) begin
    out <= large_vec[0];
  end

endmodule
