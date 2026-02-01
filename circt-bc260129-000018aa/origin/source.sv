module top;
  NestedA inst_a();
endmodule

module NestedA;
  module NestedB;
    module NestedC;
      logic [7:0] data;
      
      function automatic bit func2(input bit y);
        func2 = ~y;
      endfunction
      
      function automatic bit func1(input bit x);
        func1 = func2(x);
      endfunction
      
      always_comb data[0] = func1(data[7]);
    endmodule
    
    NestedC inst_c();
  endmodule
  
  NestedB inst_b();
endmodule