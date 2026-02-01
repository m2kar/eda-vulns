module top();
  logic [7:0] sig;
  
  always_comb begin
    for(int i=0; i<4; i++) begin
      sig[i] = i[0];
    end
  end
endmodule