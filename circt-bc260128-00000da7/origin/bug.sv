// Minimized testcase: arcilator timeout on dynamic-indexed packed struct array
// Bug: always_comb with write-then-read on same dynamic index causes infinite loop

module bug(
  input logic [2:0] idx,
  output logic result
);

  typedef struct packed {
    logic valid;
    logic [7:0] data;
  } elem_t;

  elem_t arr [0:7];

  always_comb begin
    arr[idx].data = 8'hFF;       // Write to .data field
    result = arr[idx].valid;     // Read from .valid field (same index)
  end

endmodule
