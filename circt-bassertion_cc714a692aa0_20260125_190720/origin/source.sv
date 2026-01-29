module top_module(input logic clk, input logic D, output logic Q);

  typedef struct packed {
    logic [4:0] field0;
  } array_elem_t;

  array_elem_t [7:0] data_array;

  always_comb begin
    data_array[0].field0[0] = D;
  end

  always_ff @(posedge clk) begin
    Q <= data_array[0].field0[0];
  end

endmodule