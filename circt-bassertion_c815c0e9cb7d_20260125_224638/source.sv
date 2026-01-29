module top_module(input clk, output string str_out);
  string str;
  
`ifdef ENABLE
  always @(posedge clk) begin
    str <= "Hello";
  end
  
  assign str_out = str;
`else
  assign str_out = "Default";
`endif

endmodule