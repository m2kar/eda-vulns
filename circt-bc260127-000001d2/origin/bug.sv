module m(input q);
  always_comb assert(q) else $error("");
endmodule
