module test_module(
  input logic clk,
  input logic a,
  output logic out
);
  typedef struct packed {
    logic field1;
    logic field2;
  } my_struct_t;
  my_struct_t my_struct;
  logic q;
  always_ff @(posedge clk) begin
    q <= a;
  end
  always_comb begin
    my_struct.field2 = q;
  end
  assign out = my_struct.field1;
endmodule
