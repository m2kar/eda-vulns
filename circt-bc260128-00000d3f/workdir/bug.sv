// Minimal test case: packed union as module port crashes circt-verilog
typedef union packed { logic a; } u;
module m(input u i);
endmodule
