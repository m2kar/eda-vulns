module top_module;
  logic clk, sub_out;
  logic [7:0] data_reg_data;
  logic data_reg_valid;
  
  submodule inst (
    .clk(clk),
    .sig(data_reg_valid),
    .out(sub_out)
  );
  
  always_comb begin
    data_reg_data = sub_out ? 8'hFF : 8'h00;
  end
  
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
