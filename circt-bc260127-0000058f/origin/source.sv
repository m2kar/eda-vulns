typedef union packed {
  logic [7:0] byte_data;
  logic [3:0][1:0] nibble_pairs;
} my_union;

module union_register_module(
  input logic clk,
  input logic rst_n,
  input my_union data_in,
  output my_union data_out
);

  my_union data_reg;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      data_reg <= '0;
    else
      data_reg <= data_in;
  end

  assign data_out = data_reg;

endmodule