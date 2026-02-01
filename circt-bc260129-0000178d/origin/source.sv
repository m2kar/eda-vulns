module top(input logic clk, output string out);
  string a = "Test";
  string x;
  
  always @(posedge clk) begin
    x = a;
    out = x;
  end
endmodule