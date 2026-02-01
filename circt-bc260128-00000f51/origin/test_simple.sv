module top_module;
  logic clk;
  
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end
endmodule
