module top_module;
  logic sub_out;
  
  struct packed {
    logic data;
    logic valid;
  } data_reg;
  
  submodule inst (
    .sig(data_reg.valid),
    .out(sub_out)
  );
  
  always_comb begin
    data_reg.data = sub_out;
  end
endmodule

module submodule(
  input logic sig,
  output logic out
);
  assign out = sig;
endmodule
