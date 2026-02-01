module test(input logic [7:0] x);
  always_comb begin
    assert (x == 8'h01) else $error("Assertion failed: x=%0d", x);
  end
endmodule
