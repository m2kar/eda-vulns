module test(input string a, output int b);
  wire [31:0] len_signal;
  wire [31:0] inverted_len;
  
  assign len_signal = a.len();
  assign inverted_len = ~len_signal;
  assign b = inverted_len;
endmodule