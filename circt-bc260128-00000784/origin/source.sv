module test_module(input logic clk);
  logic [3:0] idx;
  logic [7:0] arr;
  
  // Counter to update index
  always_ff @(posedge clk) begin
    idx <= idx + 1;
  end
  
  // Array assignment using counter index
  always_comb begin
    arr = 8'b0;
    arr[idx] = 1'b1;
    
    // Immediate assertion inside procedural block
    assert (arr[idx] == 1'b1) else $error("Assertion failed: arr[%0d] != 1", idx);
  end
endmodule