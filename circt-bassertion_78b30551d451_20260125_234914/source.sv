module top(input string a, output logic [7:0] out);
  int length;
  logic [7:0] in = 8'hFF;

  always_comb begin
    length = a.len();
  end
  
  assign out[7-:4] = in;
endmodule