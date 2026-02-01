module M(
  input logic enable,
  input logic [7:0] data_in,
  output logic data_out
);

  // Struct declaration with fields used in combinational logic
  typedef struct packed {
    logic valid;
    logic [7:0] value;
  } reg_t;
  
  reg_t my_reg;
  
  // Signal derived from combinational logic for assertion checking
  logic q;
  assign q = my_reg.value[0];
  
  // Procedural assignment to struct fields
  always_comb begin
    my_reg.valid = enable;
    my_reg.value = data_in;
  end
  
  // Combinational logic to assign struct field to output
  assign data_out = my_reg.valid & my_reg.value[0];
  
  // Immediate assertion inside a procedural block
  always @(*) begin
    assert (q == 1'b0) else $error("Assertion failed: q != 0");
  end

endmodule