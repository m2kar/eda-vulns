module array_example(
  input logic [2:0] idx,
  input logic clk,
  output logic result
);

  typedef struct packed {
    logic valid;
    logic [7:0] data;
  } array_elem_t;

  array_elem_t arr [0:7];

  always_comb begin
    arr[idx].data = 8'hFF;
    
    if (arr[idx].valid) begin
      result = 1'b1;
    end else begin
      result = 1'b0;
    end
  end

endmodule