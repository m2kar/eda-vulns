module top(input logic clk);
  logic [3:0] idx;
  logic q;
  
  always_ff @(posedge clk) begin
    q <= ~q;
    idx <= idx + 1;
  end
  
  assert property (@(posedge clk) q |-> idx != 0) 
    else $error("Assertion failed");
    
endmodule