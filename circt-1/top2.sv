// CIRCT Vulnerability Test Case - WORKAROUND CODE
// This file demonstrates the workaround using intermediate wire assignments
// Expected Result: Compilation SUCCEEDS

module top_arc(clkin_data, in_data, out_data);
  reg [5:0] _00_;
  input [63:0] clkin_data;
  wire [63:0] clkin_data;
  input [191:0] in_data;
  wire [191:0] in_data;
  output [191:0] out_data;
  wire [191:0] out_data;

  // WORKAROUND: Use intermediate wire assignments
  // Extract array elements to separate wires before using in sensitivity list
  wire clkin_0 = clkin_data[0];
  wire rst = clkin_data[32];
  
  always_ff @(posedge clkin_0) begin
    if (!rst) _00_ <= 6'h00;
    else _00_ <= in_data[7:2];
  end

  assign out_data[5:0] = _00_;
  assign out_data[191:6] = 186'h0;
endmodule
