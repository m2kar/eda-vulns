`define WIDTH 8

module MixedPorts(
  input logic a,
  output logic b,
  inout wire c,
  input logic [`WIDTH-1:0] data_in
);

  // Signal assignment using the macro-defined width port
  always_comb begin
    b = (data_in == 0);
  end

  // Tristate driver for the inout port using the input data
  assign c = a ? data_in[0] : 1'bz;

endmodule