module example_module(output logic [3:0] out);
  logic [3:0] internal_wire;
  
  assign internal_wire[3:0] = out[3:0];
  assign out = internal_wire;
endmodule