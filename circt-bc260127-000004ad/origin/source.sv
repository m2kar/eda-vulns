module my_module (
  input  logic enable,
  inout  logic io_sig
);

  logic out_val;
  
  assign out_val = enable;
  assign io_sig = (out_val) ? 1'b1 : 1'bz;

endmodule