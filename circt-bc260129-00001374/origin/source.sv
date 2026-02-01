module test_module(
  input logic clk,
  input logic rst_n
);

  logic reset;
  logic [7:0] counter;
  logic [7:0] arr [0:255];

  assign reset = ~rst_n;

  always_ff @(posedge clk) begin
    if (reset)
      counter <= 8'h00;
    else
      counter <= counter + 1;
  end

  always_comb begin
    assert (arr[counter] == 1'b1) else $error("Assertion failed: arr[%0d] != 1", counter);
  end

endmodule