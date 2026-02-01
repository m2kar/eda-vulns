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

  my_class::my_type obj;

  initial begin
    obj = new();
  end
endmodule
