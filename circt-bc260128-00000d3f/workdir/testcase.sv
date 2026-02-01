typedef union packed {
  logic [31:0] a;
  logic [31:0] b;
} my_union;

module Sub(
  input my_union data_in,
  input logic sel,
  output logic [31:0] result
);
  assign result = sel ? data_in.a : data_in.b;
endmodule

module Top;
  logic [31:0] x, y;
  logic sel;
  logic [31:0] z;
  
  Sub s({x, y}, sel, z);
endmodule