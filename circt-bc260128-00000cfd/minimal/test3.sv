// Minimal test case 3: Using $display instead of $error
module test_display(
  input logic clk
);

initial begin
  // Use display instead of error
  $display("Display message");
end

endmodule
