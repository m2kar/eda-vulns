module test_module(
    input logic clk,
    input logic [7:0] data_in,
    output logic [7:0] out_unpacked,
    output logic [7:0] out_packed
);

  parameter int DATA_WIDTH = 8;
  
  logic [7:0] packed_arr [0:7];
  logic [7:0] unpacked_arr [0:7];
  logic [DATA_WIDTH-1:0] data;
  int idx;

  always_ff @(posedge clk) begin
    data <= data_in;
    idx <= 0;
    
    if (data == 0) begin
      unpacked_arr[0] <= 8'hFF;
    end else begin
      unpacked_arr[0] <= data;
    end
    
    // Initialize arrays
    for (int i = 1; i < 8; i++) begin
      unpacked_arr[i] <= data + i;
      packed_arr[i] <= data + i;
    end
    packed_arr[0] <= data;
  end

  assign out_unpacked = unpacked_arr[idx];
  assign out_packed = packed_arr[idx];

endmodule