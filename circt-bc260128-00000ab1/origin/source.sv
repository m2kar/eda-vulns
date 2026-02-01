module test_module(
  input logic [1:0] vec_in,
  input string str_in,
  output int int_out
);

  logic [31:0] temp;
  
  assign temp = vec_in + str_in.len();
  
  always_comb begin
    if (vec_in == 2'b00) begin
      int_out = 0;
    end else begin
      int_out = temp + 1;
    end
    
    if (str_in == "test") begin
      int_out = 100;
    end
  end

endmodule