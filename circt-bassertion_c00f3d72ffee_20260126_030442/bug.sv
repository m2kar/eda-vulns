// Minimized test case for Arc inout port assertion failure
// Original: 24 lines -> Minimized: 6 lines
// Key pattern: inout port causes llhd::RefType which Arc cannot handle
module InoutBug(
  inout logic [7:0] data_bus
);
endmodule
