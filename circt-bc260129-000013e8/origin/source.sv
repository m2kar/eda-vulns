module MixPorts(
  input  logic signed [7:0] a,
  input  logic signed [7:0] b,
  output logic signed [7:0] result,
  inout  wire  signed [7:0] io_bus
);

  logic signed [7:0] c_reg;
  
  always_comb begin
    c_reg = a + b;
  end
  
  assign result = c_reg;
  assign io_bus = (c_reg > 0) ? c_reg : 8'bz;

endmodule