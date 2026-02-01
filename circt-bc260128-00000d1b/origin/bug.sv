// Minimal test case for CIRCT crash in extractConcatToConcatExtract
// Bug: Assertion `op->use_empty() && "expected 'op' to have no uses"` failed
// Triggered by: Array element write followed by read in mux condition
// Original CIRCT version: circt-1.139.0
// Status: Fixed in later versions (LLVM 22 tools)
module example_module(input logic a, output logic [5:0] b);
  logic [7:0] arr;
  always_comb begin
    arr[0] = a;
  end
  assign b = arr[0] ? a : 6'b0;
endmodule
