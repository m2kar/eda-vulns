module test_module(
  input logic clk,
  input logic rstn,
  input string str_var
);

  string a;
  int length;

  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      a = "test_string";
      length = 0;
    end else begin
      a = str_var;
      length = a.len();
    end
  end

endmodule