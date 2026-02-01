module test(input logic clk);
  logic [7:0] arr;
  typedef struct packed { logic valid; logic [7:0] data; } pkt_t;
  pkt_t pkt;
  logic q;
  always_comb pkt.data = arr;
  always_ff @(posedge clk) q <= pkt.valid;
endmodule
