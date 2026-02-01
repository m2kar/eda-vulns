module top_module;
  logic sub_out;
  
  struct packed {
    logic [7:0] data;
    logic valid;
  } data_reg;
  
  submodule inst (
    .sig(data_reg.valid),
    .out(sub_out)
  );
  
  always_comb begin
    data_reg.data = sub_out ? 8'hFF : 8'h00;
  end
endmodule

module submodule(
  input logic sig,
  output logic out
);
  assign out = sig;
endmodule
