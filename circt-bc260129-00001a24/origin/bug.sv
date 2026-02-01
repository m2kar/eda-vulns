module bug(
  input real in_real,
  input logic clk,
  output real out_real
);

  always_ff @(posedge clk) begin
    out_real <= in_real * 0.9;
  end

endmodule
