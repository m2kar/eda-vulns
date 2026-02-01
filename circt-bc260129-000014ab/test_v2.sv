module top(input logic clk);
  class my_class;
    function new();
    endfunction
  endclass
  
  my_class mc_handle;
  
  always @(posedge clk) begin
    my_class mc;
    mc = new();
    mc_handle = mc;
  end
endmodule
