module test(input string a, output int b);
  logic [7:0] arr [0:3];
  int idx;
  
  always_comb begin
    idx = a.len();
    arr[idx] = 8'hFF;
    b = idx;
  end
endmodule