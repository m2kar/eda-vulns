module MixedPorts(
  input  logic       clk,
  input  logic       a,
  output logic       b,
  inout  logic [7:0] c,
  output reg   [7:0] count
);

  logic drive_enable;

  // Tri-state driver for inout port
  assign c = drive_enable ? count : 8'bz;

  // Sequential logic: counter increment on clock edge
  always_ff @(posedge clk) begin
    count <= count + 1;
  end

  // Combinational logic: drive enable control based on input
  always_comb begin
    drive_enable = a;
    b = a;
  end

endmodule