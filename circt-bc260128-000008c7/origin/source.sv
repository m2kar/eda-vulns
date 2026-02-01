module test_module (
  input logic clk,
  input logic rst,
  output logic valid
);

  typedef struct packed {
    logic [7:0] data;
    logic [3:0] id;
  } pkt_t;

  pkt_t [3:0] packet_array;

  always_ff @(posedge clk) begin
    if (rst) begin
      valid <= 1'b0;
      for (int i = 0; i < 4; i++) begin
        packet_array[i].id = i;
        packet_array[i].data = 8'b0;
      end
    end else begin
      valid <= |packet_array[0].data;
    end
  end

endmodule