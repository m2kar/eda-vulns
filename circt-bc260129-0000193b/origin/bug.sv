module test(
  input logic clk,
  input real in_real,
  output real out_real
);
  always @(posedge clk) begin
    out_real <= in_real;
  end
endmodule
