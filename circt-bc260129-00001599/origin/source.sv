module MixedPorts(input logic a, output logic b, inout wire c);
  logic [3:0] temp_reg;
  
  always_comb begin
    temp_reg = 4'b0;
    for(int i=0; i<4; i++) begin
      temp_reg[i] = a;
    end
    b = temp_reg[0];
  end
endmodule