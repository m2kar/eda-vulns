module test_module(
    input logic clk,
    input logic [7:0] data_in,
    output logic [7:0] out_unpacked,
    output logic [7:0] out_packed
);
  logic [7:0] packed_arr [0:1];
  logic [7:0] unpacked_arr [0:1];

  always_ff @(posedge clk) begin
    if (data_in == 0)
      unpacked_arr[0] <= 8'hFF;
    else
      unpacked_arr[0] <= data_in;
    
    for (int i = 1; i < 2; i++)
      unpacked_arr[i] <= data_in;
    
    packed_arr[0] <= data_in;
  end

  assign out_unpacked = unpacked_arr[0];
  assign out_packed = packed_arr[0];

endmodule
