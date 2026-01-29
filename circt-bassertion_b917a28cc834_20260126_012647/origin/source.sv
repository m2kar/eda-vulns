module test_module(input logic clk, input string a);
  logic r1;
  int b;
  
  always @(posedge clk) begin
    b = a.len();
    r1 = (b > 0) ? 1'b1 : 1'b0;
  end
endmodule