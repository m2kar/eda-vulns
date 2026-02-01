package pkg;
  class container #(type T = int);
    T data;
  endclass

  class my_class;
    typedef container#(pkg::my_class) my_type;
  endclass
endpackage

module System;
  import pkg::*;

  logic logic_sig;
  logic test_signal = 1'b1;

  my_class::my_type class_obj;

  initial begin
    logic_sig = test_signal;
    class_obj = new();
  end
endmodule
