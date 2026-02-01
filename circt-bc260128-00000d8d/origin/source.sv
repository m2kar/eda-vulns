module counter(
  input logic clk,
  input logic rst,
  output logic [8:0] data_out
);

  // State type and variables
  typedef enum logic {STATE_A, STATE_B} state_t;
  state_t current_state, next_state;
  
  // 2D array for computation
  logic [1:0][1:0] temp_arr;
  
  // Array computation
  always_comb begin
    temp_arr[0][0] = (data_out[0] & data_out[1]);
  end
  
  // State transition logic
  always_comb begin
    if (temp_arr[0][0])
      next_state = STATE_A;
    else
      next_state = STATE_B;
  end
  
  // Sequential state update
  always_ff @(posedge clk or posedge rst) begin
    if (rst)
      current_state <= STATE_B;
    else
      current_state <= next_state;
  end
  
  // Counter output
  always_ff @(posedge clk or posedge rst) begin
    if (rst)
      data_out <= 9'b0;
    else
      data_out <= data_out + 1'b1;
  end

endmodule