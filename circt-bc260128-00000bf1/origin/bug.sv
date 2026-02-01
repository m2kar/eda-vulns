module m(input clk);
  typedef struct packed {logic a, b;} s;
  s x;
  logic y;
  always_comb x.a = 0;
  always_ff @(posedge clk) y <= x.b;
endmodule
