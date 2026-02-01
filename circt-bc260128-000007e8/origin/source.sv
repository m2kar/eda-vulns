module example(input logic clock, input logic d, output string str);
  logic q;
  
  always_ff @(posedge clock) begin
    q <= d;
  end
  
  always_comb begin
    if (q)
      str = "Hello";
    else
      str = "";
  end
endmodule