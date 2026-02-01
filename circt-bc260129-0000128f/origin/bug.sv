module example_module(
  input logic clk,
  input logic [3:0] length,
  input logic [31:0] data [0:15],
  output logic q
);

  logic [31:0] data_reg [0:15];

  always_ff @(posedge clk) begin
    for (int i = 0; i < length; i++) begin
      data_reg[i] <= data[i];
    end
  end

  always_ff @(posedge clk) begin
    q <= data_reg[0];
  end

endmodule
