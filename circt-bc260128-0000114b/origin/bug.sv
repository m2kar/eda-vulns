// Minimized testcase for arcilator crash in InferStateProperties pass
// Bug: cast<IntegerType> fails on packed struct type when creating hw::ConstantOp

module bug(input logic clk, input logic [7:0] in);
  typedef struct packed {
    logic [7:0] a;
  } pkt_t;

  pkt_t arr [0:1];
  logic [7:0] d;

  always_ff @(posedge clk) begin
    d <= in;
    arr[0].a <= d;
    for (int i = 1; i < 2; i++)
      arr[i].a <= arr[i-1].a;
  end
endmodule
