// Minimal testcase for circt-verilog --ir-hw timeout
// Bug: Compilation hangs when processing always_comb with bit-select
module top;
  logic [7:0] data;
  always_comb data[0] = ~data[7];
endmodule
