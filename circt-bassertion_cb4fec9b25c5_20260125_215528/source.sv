module sub_module(
  input logic clock,
  output logic [7:0] out,
  input string msg
);
  always @(posedge clock) begin
    out <= 8'h00;
  end
endmodule

module top_module (
  input logic clk,
  output logic [7:0] data
);
  string message = "Hello";
  
  sub_module inst_sub (
    .clock(clk),
    .out(data),
    .msg(message)
  );
  
  always @(posedge clk) begin
    // Minimal functionality
  end
endmodule