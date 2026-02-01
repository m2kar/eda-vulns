module M #(parameter DEPTH=16) (
  input  logic [1:0] a,
  output logic [3:0] z
);

  // Struct type definition
  typedef struct packed {
    logic [DEPTH-1:0] data;
    logic valid;
  } mystruct_t;

  // Struct signal
  mystruct_t struct_in;

  // Assignment from multi-bit input to struct field
  always_comb begin
    struct_in.data = {a, {DEPTH-2{1'b0}}};
  end

  // Instantiate parameterized module with struct input
  my_module #(.DEPTH(DEPTH)) inst (
    .D_flopped(struct_in)
  );

  // Assignment from struct to multi-bit output
  assign z = struct_in.valid ? struct_in.data[3:0] : 4'b0;

endmodule

// Parameterized module definition
module my_module #(parameter DEPTH=16) (
  input logic [DEPTH:0] D_flopped  // DEPTH bits data + 1 bit valid
);
  // Module implementation (placeholder)
endmodule