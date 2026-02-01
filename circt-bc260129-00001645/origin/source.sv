module MixedPorts(
  input  logic        clk,
  input  logic signed [15:0] a,
  input  logic        [15:0] b,
  output logic        out_b,
  inout  logic        c
);

  logic signed [15:0] signed_result;

  // Continuous assignment using both signed and unsigned ports
  assign signed_result = a + $signed(b);

  // Sequential logic with clock edge
  always_ff @(posedge clk) begin
    out_b <= signed_result[0];
  end

endmodule