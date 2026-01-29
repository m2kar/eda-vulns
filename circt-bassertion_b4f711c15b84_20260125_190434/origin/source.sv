module MixedPorts(
  input logic clk,
  input logic a,
  output logic b,
  inout wire c
);

  logic [7:0] count;

  always_ff @(posedge clk) begin
    if (a) begin
      count <= 8'd0;
    end else begin
      count <= count + 8'd1;
    end
  end

  assign b = count[0];

endmodule