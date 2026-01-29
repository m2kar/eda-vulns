module MixedPorts(
  input logic clk,
  input logic a,
  input logic dir,
  output logic b,
  inout wire c
);

  logic data_in;
  
  // Data input assignment
  assign data_in = a;
  
  // Registered output with clocked always block
  always_ff @(posedge clk) begin
    b <= data_in;
  end
  
  // Tri-state driver for inout port
  assign c = (dir) ? data_in : 1'bz;

endmodule