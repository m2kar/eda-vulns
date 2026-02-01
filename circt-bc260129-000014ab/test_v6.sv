module top;
  class my_class;
  endclass
  
  my_class mc_handle;
  
  always begin
    mc_handle = new();
  end
endmodule
