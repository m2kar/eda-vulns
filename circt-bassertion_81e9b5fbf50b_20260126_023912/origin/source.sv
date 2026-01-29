typedef union packed {
  logic [31:0] a;
  logic [31:0] b;
} my_union;

module mod1(output my_union out);
  assign out.a = 32'h1234_5678;
endmodule

module mod2(input my_union in);
  logic [31:0] val;
  assign val = in.b;
endmodule

module top();
  my_union conn;
  mod1 m1(.out(conn));
  mod2 m2(.in(conn));
endmodule