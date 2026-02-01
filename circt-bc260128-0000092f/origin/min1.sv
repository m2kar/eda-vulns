typedef struct packed {
  logic valid;
  logic [7:0] data;
} packet_t;

module array_processor(
  input logic clk,
  input logic start
);
  packet_t [3:0] packet_array;

  always_ff @(posedge clk) begin
    if (start) begin
      packet_array[0].valid <= 1'b1;
      packet_array[0].data <= 8'h20;
    end
  end
endmodule
