module test(input string a, output int b);
  logic [3:0] vec;
  
  assign b = a.len();
  
endmodule