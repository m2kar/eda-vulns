module test(input string a, output int b);
  wire [31:0] len_signal = a.len();
  wire [31:0] inverted_len;
  
  assign inverted_len = ~len_signal;
  assign b = inverted_len;
endmodule