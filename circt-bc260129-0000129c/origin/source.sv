module test_module(input logic clk, input logic rst);
  localparam signed [5:0] p = {3{((-2'sd1)-(5'd22))}};
  logic q;
  
  always_comb begin
    q = (p > 0) ? 1'b0 : 1'b1;
  end
  
  always @(*) begin
    assert (q == 1'b0) else $error("Assertion failed: q != 0");
  end
endmodule