module m(input clk, output logic out);
  logic a [0:1];
  always_ff @(posedge clk) begin
    out <= a[0];
    a[0] <= 0;
    for (int i = 1; i < 2; i++)
      a[i] <= a[i-1];
  end
endmodule
