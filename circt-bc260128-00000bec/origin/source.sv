module test_module(
  input logic clk,
  input logic D,
  output logic Q
);

  logic q_reg;

  always_ff @(posedge clk) begin
    Q <= D;
    q_reg <= Q;
  end

  always @(*) begin
    assert (q_reg == 1'b0) else $error("Assertion failed: q_reg != 0");
  end

endmodule