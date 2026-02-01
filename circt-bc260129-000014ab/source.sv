module top(input logic clk, input logic rst, input logic cond);
  
  class my_class;
    function new();
    endfunction
  endclass
  
  my_class mc_handle;
  
  always @(posedge clk) begin
    if (rst) begin
      my_class mc;
      mc = new();
      mc_handle = mc;
    end
  end
  
  always @(posedge clk) begin
    if (cond) begin
      // empty block
    end
  end
  
endmodule