typedef struct packed {
  logic valid;
  logic [7:0] data;
} packet_t;

module array_processor(
  input logic clk,
  input logic start,
  output logic done
);

  packet_t [3:0] packet_array;
  logic [2:0] counter;

  always_ff @(posedge clk) begin
    if (start) begin
      for (int i = 0; i < 4; i++) begin
        packet_array[i].valid <= 1'b1;
        packet_array[i].data <= i * 8'h20;
      end
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
