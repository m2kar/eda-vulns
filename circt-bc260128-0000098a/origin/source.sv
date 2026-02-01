module test_module(
  input logic clk,
  output logic [7:0] out
);

  logic [7:0] data_array;
  logic [7:0] arr;
  logic [2:0] idx;
  logic test_val;
  logic result_out;

  // Assignment to array element using indexing register
  always_comb begin
    arr[idx] = 1'b1;
  end

  // Assignments to connect array signals with indexed access
  always_comb begin
    data_array = arr;
    test_val = result_out;
    result_out = data_array[idx];
  end

  // Clock synchronization for sequential behavior
  always_ff @(posedge clk) begin
    out <= result_out;
  end

endmodule