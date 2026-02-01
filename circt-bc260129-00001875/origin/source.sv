module test_module(inout wire c, input logic a);
  logic [3:0] temp_reg;
  
  initial begin
    temp_reg = 4'b1010;
  end
  
  assign c = (a) ? temp_reg[0] : 1'bz;
endmodule