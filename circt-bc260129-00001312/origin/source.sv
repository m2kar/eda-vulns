module test_module(input logic clk, input logic sel, output string out_str);
  string s [0:0];
  
  always_ff @(posedge clk) begin
    if (sel)
      s[0] = "test";
    else
      s[0] = "";
  end
  
  assign out_str = s[0];
endmodule