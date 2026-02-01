module MixPorts(
  input  logic a,
  inout  wire  c
);
  assign c = a ? 1'bz : 1'b0;
endmodule
