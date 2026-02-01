module test_module(
  input logic signed [7:0] signed_data,
  output string out_str
);

  logic sel;
  string s [0:0];

  always_comb begin
    sel = (signed_data > 0);
  end

  always_comb begin
    if (sel)
      s[0] = "POS";
    else
      s[0] = "NEG";
  end

  assign out_str = s[0];

endmodule