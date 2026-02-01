typedef struct packed {
  logic valid;
  logic [7:0] data;
} pkt_t;

module test_module(
  input logic clk,
  output string result
);
  pkt_t pkt;
  string s[1];
  
  always @(posedge clk) begin
    pkt.valid <= 1'b1;
    pkt.data <= 8'hAA;
    s[0] = "hello";
    $display(":assert: ('%s' == 'hello')", s[0]);
  end
  
  assign result = (pkt.valid) ? "passed" : "failed";
  
endmodule