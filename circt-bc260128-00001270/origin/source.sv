module combined_mod(
  input logic signed [7:0] in,
  input logic [63:0] wide_input,
  output logic [31:0] out_val,
  inout logic io_sig,
  output logic signed [7:0] out
);

  // Shift operation inside a loop
  always_comb begin
    for (int i = 0; i < 8; i++) begin
      out = in << i;
    end
  end

  // Connect output value from wide input
  assign out_val = wide_input[31:0];

  // Tri-state buffer for inout port
  assign io_sig = (wide_input[0]) ? out[0] : 1'bz;

endmodule