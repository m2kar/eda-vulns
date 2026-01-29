typedef struct packed {
  logic [3:0] field1;
  logic valid;
} my_struct_t;

module MixedPorts(input logic a, output logic b, inout wire c);
  my_struct_t data;

  always_comb begin
    data.field1 = 4'b1100;
    data.valid = a;
  end

  assign b = data.valid;
endmodule