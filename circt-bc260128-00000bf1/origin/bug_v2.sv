module test(input clk);
  typedef struct packed { logic v; logic [7:0] d; } t;
  t p;
  logic q;
  always_comb p.d = 0;
  always_ff @(posedge clk) q <= p.v;
endmodule
