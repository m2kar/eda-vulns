// CIRCT Vulnerability Test Case - VULNERABLE CODE
// This file demonstrates the vulnerability where direct array indexing
// in sensitivity lists causes compilation failure
// Expected Result: Compilation FAILS with llhd.constant_time error

module top_arc(clkin_data, in_data, out_data);
  reg [5:0] _00_;
  input [63:0] clkin_data;
  wire [63:0] clkin_data;
  input [191:0] in_data;
  wire [191:0] in_data;
  output [191:0] out_data;
  wire [191:0] out_data;

  // Direct array indexing in sensitivity list - CAUSES VULNERABILITY
  // Using clkin_data[0] as clock and clkin_data[32] as reset
  always_ff @(posedge clkin_data[0])
    if (!clkin_data[32]) _00_ <= 6'h00;
    else _00_ <= in_data[7:2];

  assign out_data[5:0] = _00_;
  assign out_data[191:6] = 186'h0;
endmodule
