// Minimized test case - CIRCT assertion failure in extractConcatToConcatExtract
// Triggers: packed struct array with bit-level indexing in always_comb block
module m(input logic D, output logic Q);
  typedef struct packed { logic [1:0] f; } t;
  t [1:0] a;
  always_comb begin
    a[0].f[0] = D;
    Q = a[0].f[0];
  end
endmodule
