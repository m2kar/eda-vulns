module test_module(
  input  logic [1:0] in,
  output logic [2:0] out
);

  assign out[0] = in[0] ^ in[1];
  assign out[2] = 1'h0;

  always @* begin
    out[1] = in[0] & in[1];
  end

endmodule
