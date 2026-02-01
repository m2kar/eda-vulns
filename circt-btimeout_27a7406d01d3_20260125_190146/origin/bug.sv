module M(output logic O);
  typedef struct packed { logic a; logic b; } S;
  S s;
  always_comb s.b = 0;
  assign O = s.a;
endmodule
