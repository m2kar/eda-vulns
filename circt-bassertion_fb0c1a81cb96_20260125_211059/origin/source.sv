module simple_counter(
  input  logic       clk,
  input  logic       reset,
  output logic [7:0] count,
  output string      status_str
);
  // Counter register
  logic [7:0] count_reg;

  // Sequential logic: counter with synchronous reset
  always_ff @(posedge clk or posedge reset) begin
    if (reset) begin
      count_reg <= 8'b0;
    end else begin
      count_reg <= count_reg + 1;
    end
  end

  // Update output count
  assign count = count_reg;

  // Combinational logic to assign status string based on count value
  always_comb begin
    if (count_reg == 8'b0) begin
      status_str = "RESET";
    end else if (count_reg > 8'd100) begin
      status_str = "OVERFLOW";
    end else begin
      status_str = "COUNTING";
    end
  end
endmodule