module MixPorts(
  input logic clk,
  input logic rst,
  input logic a,
  output logic b,
  inout wire c
);

  logic [31:0] data [0:3];
  logic [1:0] write_idx;
  logic [31:0] write_data;

  // Output assignment
  assign b = a;
  
  // Inout assignment
  assign c = a ? 1'bz : 1'b0;

  // Data source for array writes
  always_comb begin
    write_data = {30'h0, write_idx};
  end

  // Counter to generate array indices
  always_ff @(posedge clk) begin
    if (rst) begin
      write_idx <= 2'b00;
    end else begin
      write_idx <= write_idx + 1;
    end
  end

  // Write operation to the data array
  always_ff @(posedge clk) begin
    if (rst) begin
      for (int i = 0; i < 4; i++) begin
        data[i] <= 32'h0;
      end
    end else begin
      data[write_idx] <= write_data;
    end
  end

endmodule