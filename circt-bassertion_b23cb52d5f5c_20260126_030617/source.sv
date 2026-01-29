module test_module(input logic in);
  logic [7:0] s;
  logic x;
  
  always_comb begin
    x = in;
    s[0] = x;
  end
endmodule