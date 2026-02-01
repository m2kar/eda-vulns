module test_module(input logic clk, input logic rst);

  // Class declaration with properties
  class my_class;
    logic [7:0] data;
    function void set_data(logic [7:0] val);
      data = val;
    endfunction
  endclass

  // Signal connecting combinational and sequential logic
  logic [7:0] computed_value;
  
  // Class object for sequential logic
  my_class mc;

  // Combinational logic block
  always @(*) begin
    computed_value = 8'hAA;  // Example combinational logic
  end

  // Sequential logic block with class instantiation
  always @(posedge clk) begin
    if (rst) begin
      mc = new();
      mc.set_data(8'h00);
    end else begin
      mc = new();
      mc.set_data(computed_value);
    end
  end

endmodule