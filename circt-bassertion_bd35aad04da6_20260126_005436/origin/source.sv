module test(input logic clk, output string str);
  logic [7:0] a;
  wire b = a[0];
  
  always_ff @(posedge clk) begin
    a <= a + 1;
  end
  
  always_comb begin
    str = "Hello";
  end
endmodule