module example(input logic clk, inout logic c);
  logic [3:0] temp_reg;
  logic a;
  
  always @(posedge clk) begin
    temp_reg <= temp_reg + 1;
  end
  
  assign c = (a) ? temp_reg[0] : 1'bz;
endmodule