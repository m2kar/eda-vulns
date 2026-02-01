module top_module;
  logic clk, sub_out;
  
  struct packed {
    logic [7:0] data;
    logic valid;
  } data_reg;
  
  always_comb begin
    data_reg.data = sub_out ? 8'hFF : 8'h00;
  end
  
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
