`define ENABLE

module MixedPorts(
  input logic clk,
  input logic a,
  output logic b,
  inout logic c
);

  // Continuous assignment connecting inout port to internal signal
  assign c = a;

`ifdef ENABLE
  always_ff @(posedge clk) begin
    b <= a;
  end
`endif

endmodule