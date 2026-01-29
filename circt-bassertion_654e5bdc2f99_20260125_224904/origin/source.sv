module MixedPorts(
  input  signed [5:0] a,
  input  signed [5:0] b,
  output signed [5:0] c,
  inout  signed [5:0] d
);

  // Signed right shift operation with signed variable shift amount
  assign c = a >>> b;
  
  // Inout port connection - drive d based on b[0]
  assign d = (b[0]) ? c : a;

endmodule