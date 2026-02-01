module MixedPorts(
  input logic clk,
  input logic a,
  output logic b,
  inout logic c
);

  logic [3:0] counter = 0;
  
  always_ff @(posedge clk) begin
    if (a) begin
      counter <= counter + 1;
    end
  end
  
  assign b = (counter == 4'b1111);
  
  assign c = (counter[0]) ? 1'b1 : 1'bz;

endmodule