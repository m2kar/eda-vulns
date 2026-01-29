module string_register(
  input logic clk,
  input string data_in,
  output string data_out
);

  string a = "Test";
  
  assign data_in = a;
  
  always @(posedge clk) begin
    data_out <= data_in;
  end

endmodule