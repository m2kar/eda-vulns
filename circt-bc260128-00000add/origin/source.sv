module test_module (inout logic io_sig);
  logic [1:0] out_val;
  
  always_comb begin
    out_val = 2'b01;
  end
  
  assign io_sig = (out_val[0]) ? 1'b1 : 1'bz;
endmodule