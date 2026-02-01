module test_module(
  input logic clk,
  input logic [7:0] clkin_data,
  output string s[1:0]
);

  // Packed array for clock data (already provided as input)
  
  // Initialize string variables
  initial begin
    s[0] = "hello";
    s[1] = "world";
  end
  
  // Always_ff block triggered by bit-select of clkin_data as clock edge
  always_ff @(posedge clkin_data[0]) begin
    s[1] <= "triggered";
  end

endmodule