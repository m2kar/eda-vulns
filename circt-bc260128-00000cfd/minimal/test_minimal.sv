module test_minimal(
  input logic clk
);

initial begin
  assert (1'b0 == 1'b1) else $error("Assertion failed");
end

endmodule
