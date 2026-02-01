module top(input logic clk);
  class my_class;
  endclass
  
  my_class mc_handle;
  
  always @(posedge clk) begin
    mc_handle = new();
  end
endmodule
