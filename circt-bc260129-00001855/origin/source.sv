module test_module (
  input logic clk,
  output logic out,
  inout logic port_a,
  inout logic port_b
);

  logic sig;
  
  assign port_a = sig;
  assign port_b = sig;
  
  always @(posedge clk) begin
    sig = out;
  end
  
  always_comb begin
    out = ~sig;
  end

endmodule