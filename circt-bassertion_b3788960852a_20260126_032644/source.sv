module test(input signed [5:0] a, output signed [3:0] b, output string msg);
  // String variable with initialization
  string str_var = "Test";

  // Signed arithmetic with width truncation/extension
  assign b = a / 4'sd2;

  // Continuous assignment of string output port
  assign msg = str_var;

  initial begin
    $display("b=%0d, msg=%s", b, msg);
  end
endmodule