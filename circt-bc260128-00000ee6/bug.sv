module m(
  output logic [3:0] a
);

  logic w = 1'b0;

  always_comb begin
    a[0] = w;
    a[1] = w;
  end

endmodule
