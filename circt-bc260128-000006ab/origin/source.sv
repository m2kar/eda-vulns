module MixPorts(
  input  logic a,
  output logic b,
  inout  wire  c
);
  parameter int P1 = 8;
  
  logic [P1-1:0] counter;
  
  assign b = a;
  assign c = a ? 1'bz : 1'b0;
  
  always_comb begin
    for (int i = 0; i < P1; i++) begin
      counter[i] = i[0];
    end
  end
endmodule