module test_module(
  input logic clk,
  output logic [7:0] result
);

  // Array declaration and index variable
  logic [7:0] arr [0:7];
  logic [2:0] idx = 3'b0;

  // Function that increments a value
  function automatic [7:0] add_one(input [7:0] val);
    add_one = val + 1;
  endfunction

  // Combinational logic with function call
  always_comb begin
    result = add_one(8'h5);
  end

  // Immediate assertion with array access
  always @(*) begin
    assert (arr[idx] == 1'b1) else $error("Assertion failed: arr[%0d] != 1", idx);
  end

  // Clocked process to update array values
  always_ff @(posedge clk) begin
    arr[idx] <= result[0];
    idx <= idx + 1;
  end

endmodule