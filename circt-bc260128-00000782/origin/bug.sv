module m(input x);
  always_comb assert (x) else $error("f");
endmodule
