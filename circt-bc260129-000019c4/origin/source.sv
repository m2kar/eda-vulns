module top_module (
  inout wire my_pin
);

  logic [7:0] data_array [0:3];
  
  initial begin
    data_array <= '{default: 8'hFF};
  end
  
  assign my_pin = data_array[0][0] ? 1'bz : 1'b0;

endmodule