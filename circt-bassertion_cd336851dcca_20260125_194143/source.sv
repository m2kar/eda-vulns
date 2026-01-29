module test_module(input logic clk, output string result);
  logic r1;
  
  always @(posedge clk) begin
    r1 = 0;
  end
  
  function string process_string(string s = "");
    return s;
  endfunction
  
  assign result = process_string("");
endmodule