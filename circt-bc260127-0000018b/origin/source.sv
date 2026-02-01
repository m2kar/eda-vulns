module test_module;
  logic [7:0] arr = 8'h0;
  int idx = 0;
  
  always_comb begin
    arr[idx] = 1'b1;
  end
endmodule