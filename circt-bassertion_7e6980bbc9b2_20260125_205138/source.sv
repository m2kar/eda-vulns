module top(
  input  logic clk_data,
  input  logic A,
  input  logic C,
  output logic O,
  output string str_out
);

  string str;

  always_ff @(posedge clk_data) begin
    if (C)
      O <= A;
  end

  always_comb begin
    str = (A) ? "high" : "low";
  end

  assign str_out = str;

endmodule