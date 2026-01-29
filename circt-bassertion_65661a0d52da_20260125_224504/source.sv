module Mod(input logic [1:0] a, output string str_out);
  string str = "test";

  always_comb begin
    if (a == 2'b00)
      str_out = str;
    else
      str_out = "default";
  end
endmodule

module Top;
  string my_str;
  Mod inst1(.a(2'b01), .str_out(my_str));
endmodule