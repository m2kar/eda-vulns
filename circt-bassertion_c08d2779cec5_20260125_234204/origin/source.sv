module test_module(input logic in, output string str_out);
  string str;
  logic x;
  
  always_comb begin
    str = "test";
    x = in;
  end
  
  assign str_out = str;
endmodule