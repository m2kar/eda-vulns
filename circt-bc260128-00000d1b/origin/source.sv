module example_module(input logic a, output logic [5:0] b);
  logic [7:0] arr;
  
  always_comb begin
    arr[0] = a;
  end
  
  assign b = arr[0] ? a : 6'b0;
endmodule