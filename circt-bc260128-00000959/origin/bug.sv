module test;
  always @(*) begin
    assert (1'b0) else $error("fail");
  end
endmodule
