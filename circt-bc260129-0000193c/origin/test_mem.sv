module test_module(
  input logic clk,
  input logic [19:0] addr,
  output logic [16777215:0] data_out
);

  logic [16777215:0] memory [0:255];
  
  always @(posedge clk) begin
    data_out <= memory[addr];
  end

endmodule
