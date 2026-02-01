module top_module;
  logic clk, sub_out;
  
  submodule inst (
    .clk(clk),
    .sig(clk),
    .out(sub_out)
  );
  
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end
endmodule

module submodule(
  input logic clk,
  input logic sig,
  output logic out
);
  always_ff @(posedge clk) begin
    out <= sig;
  end
endmodule
