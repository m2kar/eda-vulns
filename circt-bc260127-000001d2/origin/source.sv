module test_module(input logic clk, output logic q);
  
  // Signal assignment to drive q (toggle on clock edge)
  always_ff @(posedge clk) begin
    q <= ~q;
  end
  
  // Immediate assertion inside a procedural block
  always_comb begin
    assert (q == 1'b0) else $error("Assertion failed: q != 0");
  end

endmodule