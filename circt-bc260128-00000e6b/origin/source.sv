module test(input string a, output int b);
  logic temp;
  
  always_comb begin
    temp = (a.len() > 0);
  end
  
  assign b = temp ? 1 : 0;
endmodule