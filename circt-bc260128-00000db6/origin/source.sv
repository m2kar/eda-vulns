module array_example(input logic a, input logic b);
  logic [1:0] arr;
  
  assign arr[0] = a;
  
  always_comb arr[1] = b;
endmodule