module top_module(input logic clk, input logic resetn);
  
  // Class definition with typedef referencing itself in parameterized template
  class registry #(type T = int);
    // Parameterized registry class
  endclass
  
  class my_class;
    typedef registry#(my_class) type_id;
  endclass
  
  // Instance of the class
  my_class obj;
  
  // Sequential logic with clock edge
  always_ff @(posedge clk) begin
    if (resetn) begin
      obj = new();
    end
  end
  
endmodule