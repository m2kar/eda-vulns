module top(input logic clk);
  class C; endclass
  C obj;
  always_ff @(posedge clk) obj = new();
endmodule
