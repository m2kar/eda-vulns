typedef struct packed {
  logic valid;
} packet_t;

module array_processor(
  input logic clk,
  input logic start,
  output logic done
);

  packet_t packet_array [0:0];
  logic counter;

  always_ff @(posedge clk) begin
    if (start) begin
      packet_array[0].valid <= 1'b1;
    end
  end

  always_ff @(posedge clk) begin
    if (start)
      counter <= 1'b0;
    else
      counter <= ~counter;
  end

  assign done = counter;

endmodule
