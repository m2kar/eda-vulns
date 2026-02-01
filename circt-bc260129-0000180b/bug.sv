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

  my_class::my_type class_obj;

  initial begin
    class_obj = new();
    #1;
  end
endmodule
