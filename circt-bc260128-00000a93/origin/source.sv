module test(input string a, output int b);
  logic [31:0] shared_signal;
  
  assign shared_signal = a.len();
  assign b = shared_signal;
endmodule