module MixedPorts(
  input logic a,
  output logic b,
  inout logic c,
  input logic clk
);

  logic c_drive;

  always_ff @(posedge clk) begin
    b <= a;
    c_drive <= a;
  end

  assign c = c_drive;

endmodule