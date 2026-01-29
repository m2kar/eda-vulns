module mixed_assignments(
  input logic clk,
  input logic [7:0] P1,
  input logic [7:0] P2,
  output string str_out
);

  string str;
  logic [7:0] r1;

  always @(posedge clk) begin
    r1 = P1;
    r1 <= P2 + r1;
  end

  always_comb begin
    if (r1 > 100)
      str = "HIGH";
    else
      str = "LOW";
  end

  assign str_out = str;

endmodule