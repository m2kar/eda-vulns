module M(
  input  logic A,
  output logic O
);

  // Packed struct declaration with multiple fields
  typedef struct packed {
    logic field1;
    logic field2;
  } my_struct_t;

  // Packed struct variable declaration
  my_struct_t my_struct;

  // Wire for conditional
  logic C;

  // Assignment to packed struct field
  always_comb begin
    my_struct.field2 = 1'b0;
  end

  // Member access to packed struct field used in conditional
  assign C = my_struct.field1;

  // Conditional statement (if-else) inside always block
  always_comb begin
    if (C)
      O = A;
    else
      O = 1'b1;
  end

endmodule