module test(input logic in, output logic out, output string str);
  logic x;
  
  always_comb begin
    x = in;
    out = x;
  end
  
  initial begin
    str = "Hello";
  end
endmodule