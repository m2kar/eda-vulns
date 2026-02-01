typedef union packed {
  logic [31:0] a;
  logic [31:0] b;
} my_union;

module top(output logic q, output logic ok, output my_union data);
  my_union u_data;
  
  assign u_data.a = 32'hdeadbeef;
  
  always_comb begin
    if (u_data.b == 32'hdeadbeef) begin
      q = 1'b1;
      ok = 1'b1;
    end else begin
      q = 1'b0;
      ok = 1'b0;
    end
  end
  
  assign data = u_data;
endmodule