// Minimal test case 1: Just the assertion with format string
module test_assertion(
  input logic clk
);

initial begin
  // Simple assertion with format string
  assert (1'b0 == 1'b1) else $error("Assertion failed");
end

endmodule
