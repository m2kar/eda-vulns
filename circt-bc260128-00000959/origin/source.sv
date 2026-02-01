module test_module (
  input logic clk,
  input logic rst
);
  // Unpacked array declaration
  wire [31:0] arr[2];
  
  // Assignment to unpacked array elements
  assign arr[0] = 32'h00000000;
  assign arr[1] = 32'h00000001;
  
  // Signal derived from the unpacked array for assertion
  logic q;
  
  // Assignment to derive q from array content
  assign q = (arr[0][0] | arr[1][0]);
  
  // Immediate assertion inside a procedural block
  always @(*) begin
    assert (q == 1'b0) else $error("Assertion failed: q != 0");
  end
endmodule