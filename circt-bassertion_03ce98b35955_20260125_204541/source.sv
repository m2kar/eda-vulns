typedef union packed {
  logic [31:0] a;
  logic [31:0] b;
} my_union;

module Sub(input my_union in_val, output my_union out_val);
  assign out_val = in_val;
endmodule

module Top;
  my_union data_in, data_out;
  
  Sub s(.in_val(data_in), .out_val(data_out));
endmodule