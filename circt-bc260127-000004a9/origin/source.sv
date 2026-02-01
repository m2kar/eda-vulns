module MixPorts(
  input  logic [63:0] wide_input,
  output logic [31:0] out_val,
  inout  logic        io_sig
);

  logic [63:0] arg0;

  assign arg0 = wide_input;
  assign out_val = arg0[31:0];
  assign io_sig = (out_val[0]) ? 1'b1 : 1'bz;

endmodule