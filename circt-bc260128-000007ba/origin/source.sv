module MixPorts(
  input logic clk,
  input logic [63:0] wide_input,
  output logic [31:0] out_val,
  inout logic io_sig
);

  integer idx = 0;

  always_ff @(posedge clk) begin
    idx <= idx + 1;
  end

  always_comb begin
    out_val = wide_input[(idx%32)+:32];
  end

endmodule