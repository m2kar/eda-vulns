// Minimized test case for CIRCT bug: real type in sequential logic
// Original crash: integer bitwidth is limited to 16777215 bits
// Actual error: hw.bitcast with invalid bitwidth (i1073741823 -> f64)
module test(input logic clk);
  real r;
  always_ff @(posedge clk) r <= 1.0;
endmodule
