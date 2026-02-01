module m(input a,b);
  logic [1:0] x;
  assign x[0] = a;
  always_comb x[1] = b;
endmodule
