typedef union packed {
  logic [15:0] a;
  logic [15:0] b;
} union_t;

module packet_processor(
  input union_t in_packet,
  output logic [15:0] result
);
  function logic [15:0] process_union(input logic [15:0] val);
    return val + 1;
  endfunction

  assign result = process_union(in_packet.a);
endmodule

module top;
  union_t u1;
  logic [15:0] res;
  
  packet_processor proc(u1, res);
  
  initial begin
    u1.a = 16'h1234;
    #1;
    $display("Result: %0d", res);
  end
endmodule