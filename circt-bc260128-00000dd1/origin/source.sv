module MixPorts(
  input logic clk,
  input logic [63:0] wide_input,
  output logic [31:0] out_val,
  inout logic io_sig
);

  logic [31:0] data_array [0:1023];
  
  always @(posedge clk) begin
    out_val <= data_array[wide_input[9:0]];
  end
  
  assign io_sig = data_array[0][0];

endmodule