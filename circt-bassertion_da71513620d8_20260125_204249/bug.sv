module test_module(
  input logic clk,
  input logic [31:0] data_in,
  output string str_out
);

  string str;

  always_comb begin
    str = "Hello";
  end

  assign str_out = str;

endmodule
