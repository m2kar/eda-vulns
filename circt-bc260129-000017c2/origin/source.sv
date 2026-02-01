module MixedPorts(
  input  logic clk,
  input  logic a,
  output logic b,
  inout  logic c
);

  logic sig;
  logic c_drive;
  
  // Inout port bidirectional behavior
  assign c = c_drive;
  
  // Shared source signal assignment from inout driver
  assign sig = c_drive;
  
  // Same source signal driving multiple destinations
  always_ff @(posedge clk) begin
    a <= sig;
    b <= sig;
  end

endmodule