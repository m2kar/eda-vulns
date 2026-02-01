// CIRCT Bug: circt-verilog hangs on self-referential struct output in always_comb
// Testcase ID: 260128-0000074e

typedef struct packed { logic a; logic b; } s_t;

module top(output s_t out);
  always_comb begin
    out.b = out.a;  // Read from out.a, write to out.b -> causes infinite loop
  end
endmodule
