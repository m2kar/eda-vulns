module m(input c, output logic q);
  always @(negedge c) q <= 0;
  assign q = q;
endmodule
