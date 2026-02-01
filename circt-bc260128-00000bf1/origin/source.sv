module test(
  input logic clk
);
  // Array declaration
  logic [7:0] arr [0:3];
  
  // Struct type declaration
  typedef struct packed {
    logic valid;
    logic [7:0] data;
  } packet_t;
  
  packet_t pkt;
  
  // Registered signal
  logic q;
  
  // Dataflow: connect array element to struct data field
  always_comb begin
    pkt.data = arr[0];
  end
  
  // Sequential assignment: connect struct field to registered signal
  always_ff @(posedge clk) begin
    q <= pkt.valid;
  end
endmodule