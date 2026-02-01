module M(input logic clk, input logic rst, input logic [1023:0] in, output logic [7:0][63:0] out);
  
  class my_class;
    logic [63:0] data;
    function void set_data(logic [63:0] val);
      data = val;
    endfunction
  endclass
  
  my_class mc;
  
  always_ff @(posedge clk) begin
    if (rst) begin
      mc = new();
      mc.set_data(in[63:0]);
    end
  end
  
  always_comb begin
    for (int i = 0; i < 8; i++) begin
      out[i] = in[(i+1)*64-1 -: 64];
    end
  end
  
endmodule