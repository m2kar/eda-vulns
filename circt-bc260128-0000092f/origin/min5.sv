typedef struct packed {
  logic valid;
} packet_t;

module array_processor(
  input logic clk,
  input logic start,
  output logic done
);

  packet_t packet_array [0:0];
  logic [2:0] counter;

  always_ff @(posedge clk) begin
    if (start) begin
      packet_array[0].valid <= 1'b1;
    end
  end

  always_ff @(posedge clk) begin
    if (start)
      counter <= 3'b0;
    else if (counter < 3'b100)
      counter <= counter + 1;
  end

  assign done = (counter == 3'b100);

endmodule
