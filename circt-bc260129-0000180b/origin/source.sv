package pkg;
  class container #(type T = int);
    T data;
  endclass
  
  class my_class;
    typedef container#(pkg::my_class) my_type;
  endclass
endpackage

module Sub(input logic a, output logic b);
  assign b = a;
endmodule

module System;
  import pkg::*;
  
  logic logic_sig;
  logic result_sig;
  logic test_signal = 1'b1;
  
  // Instantiate structural module with implicit named port mapping
  Sub sub_inst(.a(logic_sig), .b(result_sig));
  
  // Declare class-based type
  my_class::my_type class_obj;
  
  initial begin
    // Drive structural module
    logic_sig = test_signal;
    
    // Use class-based type from package
    class_obj = new();
    
    // Monitor structural output
    #1 $display("Result: %b", result_sig);
  end
endmodule