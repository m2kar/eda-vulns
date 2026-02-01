// Minimized test case for CIRCT arc::StateType assertion failure
// Original: 260128-00000ed6
// Trigger: inout wire with tri-state assignment creating !llhd.ref<i1>
module MixPorts(
  input  logic clk,
  input  logic a,
  inout  wire  c
);
  // Tri-state assignment to inout wire - creates !llhd.ref<i1> type
  assign c = a ? 1'bz : 1'b0;
endmodule
