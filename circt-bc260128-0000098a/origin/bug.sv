module m(output logic r);
  logic [1:0] a;
  logic i;
  always_comb begin a[i] = 1; r = a[i]; end
endmodule
