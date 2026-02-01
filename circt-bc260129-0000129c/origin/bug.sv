module test_module;
  logic q;

  always_comb begin
    q = 1'b1;
  end

  always @(*) begin
    assert (q == 1'b0) else $error("Assertion failed: q != 0");
  end
endmodule
