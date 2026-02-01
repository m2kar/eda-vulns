module MixedPorts(
  input  logic a,
  output logic b,
  inout  wire  c
);

  logic r1;

  always_comb begin
    r1 = a;
  end

  assign b = r1;

  assign c = (r1) ? 1'bz : 1'b0;

endmodule