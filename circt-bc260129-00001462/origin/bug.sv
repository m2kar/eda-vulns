module top(input logic clk);
  logic a, b;
  assert property (@(posedge clk) a |-> b) else $error("msg");
endmodule
