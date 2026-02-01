module test_module(output logic [31:0] result);
  logic [1:0][1:0] temp_arr;
  enum {STATE_A, STATE_B} current_state;
  
  always_comb begin
    temp_arr[0][0] = result[0];
  end
  
  always_comb begin
    if (temp_arr[0][0]) current_state = STATE_A;
    else current_state = STATE_B;
  end
  
  assign result = {30'b0, temp_arr[0][0], current_state == STATE_A};
endmodule