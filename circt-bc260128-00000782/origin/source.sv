module test_module(
  input logic clk,
  output logic [7:0] arr
);

  logic [2:0] idx;

  // Clocked always block that increments the counter
  always_ff @(posedge clk) begin
    idx <= idx + 1;
  end

  // Combinational logic for array assignment and assertion
  always_comb begin
    arr = 8'b0;  // Initialize array
    arr[idx] = 1'b1;
    
    // Immediate assertion
    assert (arr[idx] == 1'b1) else $error("Assertion failed: arr[%0d] != 1", idx);
  end

endmodule