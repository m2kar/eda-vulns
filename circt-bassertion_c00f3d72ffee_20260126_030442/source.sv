module MixedPorts(
  input logic clk,
  input logic [7:0] data_in,
  output logic [7:0] data_out,
  inout logic [7:0] data_bus
);

  logic [7:0] arr [0:15];
  logic [3:0] idx;
  logic direction;

  // Array element access using variable index
  assign data_out = arr[idx];

  // Bidirectional bus assignment for inout port
  assign data_bus = (direction) ? arr[idx] : 8'bz;

  // Write operation to populate the array
  always_ff @(posedge clk) begin
    arr[idx] <= data_in;
  end

endmodule
