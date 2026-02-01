module test_module(
  input logic clk,
  input logic rst,
  output logic [1:0] sel,
  output string s
);

  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      sel <= 2'b00;
    end else begin
      sel <= sel + 1'b1;
    end
  end

  always_comb begin
    case (sel)
      2'b00: s = "state0";
      2'b01: s = "state1";
      2'b10: s = "state2";
      2'b11: s = "state3";
    endcase
  end

endmodule