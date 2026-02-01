module test_module(
  input  logic [1:0] in,
  output logic [3:0] out
);

  // Continuous assignments to output bits
  assign out[0] = in[0] ^ in[1];
  assign out[3] = 1'h0;

  // Combinational always block with implicit sensitivity list
  always @* begin
    out[1] = in[0] & in[1];
    out[2] = in[0] | in[1];
  end

endmodule