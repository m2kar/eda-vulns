module test_module(
  input logic a
);

  always @(*) begin
    assert (a == 1'b0) else $error("Assertion failed: a != 0");
  end

endmodule
