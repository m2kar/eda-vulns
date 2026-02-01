module TopModule (
  input wire clk,
  input logic [63:0] wide_input,
  output reg [7:0] out,
  inout logic io_sig
);

  always @(posedge clk) begin
    out <= wide_input[7:0] ^ wide_input[15:8];
  end

  assign io_sig = (wide_input[0]) ? out[0] : 1'bz;

endmodule