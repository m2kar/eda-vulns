module array_reg(
  input logic clk,
  input logic [7:0] data_in,
  output logic [11:0] sum_out
);

  logic [7:0] arr [0:15];
  logic [11:0] sum;
  
  // Combinational sum of array elements
  always_comb begin
    sum = '0;
    for (int i = 0; i < 16; i++) begin
      sum = sum + arr[i];
    end
  end
  
  // Registered output assignment
  always_ff @(posedge clk) begin
    sum_out <= sum;
  end
  
  // Array initialization with input data (shift register)
  always_ff @(posedge clk) begin
    arr[0] <= data_in;
    for (int i = 1; i < 16; i++) begin
      arr[i] <= arr[i-1];
    end
  end

endmodule