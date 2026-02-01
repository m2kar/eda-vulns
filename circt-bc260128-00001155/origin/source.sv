module string_processor(input logic clk, input string a);
  string data_array [0:3];
  int result;
  
  always @(posedge clk) begin
    data_array[0] = a.toupper();
    result = data_array[0].len();
  end
endmodule