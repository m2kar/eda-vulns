module test_module;
  logic [15:0] arr;
  logic temp;
  logic [3:0] idx;
  
  initial begin
    idx = 4'd0;
  end
  
  always @(*) begin
    arr[idx] = 1'b1;
  end
  
  assign temp = arr[0];
  
endmodule