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

  Sub sub_inst(.a(logic_sig), .b(result_sig));

  my_class::my_type class_obj;

  initial begin
    logic_sig = 1'b1;
    class_obj = new();
  end
endmodule
