module example(
  input  logic clock,
  input  logic reset,
  input  logic enable,
  output string str
);

  logic [31:0] d;
  logic [31:0] q;

  // Register with asynchronous reset and enable condition
  always @(posedge clock, posedge reset) begin
    if (reset)
      q <= 32'd42;
    else if (enable)
      q <= d;
  end

  // Connect register output to data input (increment feedback)
  assign d = q + 1;

  // String assignment using register value
  always_comb begin
    str = $sformatf("Value: %0d", q);
  end

endmodule