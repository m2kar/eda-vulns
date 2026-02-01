module m(input clk);
  class c; endclass
  c o;
  always @(posedge clk) o = new();
endmodule
