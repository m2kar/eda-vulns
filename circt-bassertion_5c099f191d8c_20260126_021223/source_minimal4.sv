module test_module(
  input  logic [1:0] in,
  output logic [1:0] out
);

  assign out[0] = in[0] ^ in[1];
  assign out[1] = 1'h0;

endmodule
