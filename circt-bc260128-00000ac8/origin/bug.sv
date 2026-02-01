// Minimal testcase: packed struct array shift with for-loop triggers
// hw::ConstantOp type assertion failure in InferStateProperties pass
module test(
  input logic clk,
  output logic o
);
  struct packed { logic d; } s[2];

  always @(posedge clk) begin
    for (int i = 1; i < 2; i++)
      s[i] <= s[i-1];
    o <= s[1].d;
  end
endmodule
