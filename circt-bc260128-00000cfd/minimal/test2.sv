// Minimal test case 2: Without llhd.process wrapper
module test_no_llhd(
  input logic clk
);

initial begin
  // Simple assertion with format string, no procedural block
  assert (1'b0 == 1'b1) else $error("Assertion failed");
end

endmodule
