module M (output logic z);
  struct packed { logic a; logic b; } s;
  always_comb s.a = 1;
  assign z = s.b;
endmodule
