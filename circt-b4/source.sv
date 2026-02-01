module test_module(input logic [31:0] arg0, output string a);
  logic [31:0] arr [0:3];

  assign arr[0] = arg0[31:0];

  always_comb begin
    if (arr[0] != 0) begin
      a = "test";
    end else begin
      a = "";
    end
  end

  // Use string method call with a.len() to ensure feature is present
  // This is a combinational assignment to an implicit internal variable.
  int length = a.len();

endmodule