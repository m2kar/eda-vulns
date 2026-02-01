module M(input logic [1:0] a, output string str);
  always_comb begin
    case (a)
      2'd0: str = "zero";
      default: str = "other";
    endcase
  end
endmodule