module array_processor(input logic clk, output logic [7:0] result_out);
  logic [15:0] arr;
  int idx;
  
  always @(posedge clk) begin
    arr <= arr + 1;
  end
  
  always @(*) begin
    idx = 0;
    assert (arr[idx] == 1'b1) else $error("Assertion failed: arr[%0d] != 1", idx);
  end
  
  assign result_out = arr[7:0];
endmodule