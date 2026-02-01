module bug(input logic clk, output logic q);
  logic d;

  always @(negedge clk) begin
    q <= d;
  end

  assign q = q;  // Combinational loop
endmodule
