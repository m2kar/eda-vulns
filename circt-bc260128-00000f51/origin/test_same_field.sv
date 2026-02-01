module top_module;
  logic clk, sub_out;
  
  struct packed {
    logic [7:0] data;
    logic valid;
  } data_reg;
  
  submodule inst (
    .clk(clk),
    .sig(clk),
    .out(sub_out)
  );
  
  always_comb begin
    data_reg.valid = sub_out ? 1'b1 : 1'b0;
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
