module test_module(
  input logic clk,
  input logic rst,
  output string str_out
);

  string str;

  initial begin
    str = "default";
  end

  always @(posedge clk) begin
    if (!rst) begin
      str <= "reset";
    end else begin
      str <= "normal";
    end
  end

  assign str_out = str;

endmodule