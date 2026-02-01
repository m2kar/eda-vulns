module my_module(
  input logic in_signal,
  output logic result,
  output logic [3:0] result_array
);

  logic internal_wire;

  assign internal_wire = in_signal;

  always_comb begin
    result_array[0] = internal_wire;
    result = internal_wire;
  end

endmodule