module MixedPorts(
  input  logic a,
  output logic b,
  inout  logic c
);

  // Propagate input to output
  assign b = a;
  
  // Bidirectional buffer for inout port
  assign c = a ? 1'bz : 1'b0;

endmodule