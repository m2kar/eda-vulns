module top_module;
  logic clk, sub_out;
  
  always_comb begin
    sub_out = clk ? 1'b1 : 1'b0;
  end
  
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end
endmodule
