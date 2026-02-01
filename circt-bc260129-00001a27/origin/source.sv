class param_class #(parameter int WIDTH = 32);
  typedef logic [WIDTH-1:0] data_t;
endclass

module dut #(parameter int WIDTH = 8)(
  input logic [WIDTH-1:0] x, 
  output logic [WIDTH-1:0] y
);
  assign y = x;
endmodule

module top;
  localparam int BUS_WIDTH = 16;
  
  param_class #(16)::data_t test_data;
  
  logic [BUS_WIDTH-1:0] a, b;
  
  dut #(.WIDTH(BUS_WIDTH)) u_dut(.x(a), .y(b));
  
  always_comb begin
    a = test_data;
    test_data = b;
  end
endmodule