module example(output string out);
  string str;
  always_comb begin
    str = "Hello";
    out = str;
  end
endmodule