typedef struct packed {
  logic [7:0] value;
} data_struct_t;

typedef union packed {
  data_struct_t s;
  logic [7:0] raw;
} data_union_t;

module data_processor(
  input logic clk,
  input data_union_t in_data,
  output logic [7:0] result
);

  function logic [7:0] process_data(data_union_t du);
    return du.s.value ^ du.raw;
  endfunction

  always_ff @(posedge clk) begin
    result <= process_data(in_data);
  end

endmodule