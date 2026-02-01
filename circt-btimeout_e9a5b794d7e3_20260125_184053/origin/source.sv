module test_module(
  input logic clk,
  input logic cond1,
  input logic cond2,
  input logic a,
  input logic b,
  input logic c,
  output logic out
);

  // Packed struct declaration
  typedef struct packed {
    logic field1;
    logic field2;
  } my_struct_t;

  // Packed struct variable declaration and initialization
  my_struct_t my_struct = '{1'b1, 1'b0};

  // Flip-flop output declaration
  logic q;

  // Flip-flop with priority control
  always_ff @(posedge clk) begin
    if (cond1)
      q <= a;
    else if (cond2)
      q <= b;
    else
      q <= c;
  end

  // Connection to struct field
  always_comb begin
    my_struct.field2 = q;
  end

  // Member access to packed struct field
  assign out = my_struct.field1;

endmodule