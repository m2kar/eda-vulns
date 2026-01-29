module test_module(input logic clk, input logic rst_n, output string out_str);
  string a = "Test";

  // Using a variable with the same lifetime as out_str to hold the next value
  string next_out_str;

  // Initialize the output string at simulation start
  initial out_str = "";

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_str <= "";
    end else begin
      out_str <= a;
    end
  end
endmodule