module test(input logic clk, output string result);
  reg r1;
  string s;
  
  initial begin
    s = "hello";
  end
  
  always_ff @(posedge clk) begin
    r1 <= |r1;
  end
  
  always_comb begin
    result = s;
    $display(":assert: ('%s' == 'hello')", s);
  end
endmodule