module xor_shift_module(
  input logic clk,
  output logic result_out
);

  // Struct declaration with data and valid fields
  typedef struct packed {
    logic valid;
    logic [7:0] data;
  } pkt_t;

  // Struct variable and shift register
  pkt_t shift_reg[4];
  logic [7:0] xor_value;

  // Always block that shifts struct data and computes XOR
  always @(posedge clk) begin
    shift_reg[0].valid <= 1'b1;
    shift_reg[0].data <= 8'hA5;
    for (int i = 1; i < 4; i++) begin
      shift_reg[i] <= shift_reg[i-1];
    end
    xor_value <= shift_reg[3].data ^ shift_reg[2].data;
    result_out <= ^xor_value;
  end

endmodule