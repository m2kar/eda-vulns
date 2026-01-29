module example_module(
  input logic clk,
  output string out
);
  string str;
  
  always_comb begin
    str = "test";
    out = str;
  end
endmodule