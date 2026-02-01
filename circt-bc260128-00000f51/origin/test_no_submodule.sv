module top_module;
  logic sub_out;
  
  struct packed {
    logic data;
    logic valid;
  } data_reg;
  
  assign sub_out = data_reg.valid;
  
  always_comb begin
    data_reg.data = sub_out;
  end
endmodule
