module data_processor(
  input logic clk,
  input logic [31:0] in_data,
  output logic [3:0] result
);

  // Struct type definition for packet_t
  typedef struct packed {
    logic [7:0] header;
    logic [31:0] payload;
  } packet_t;

  // Unpacked array of packet_t structs
  packet_t packet_array [0:3];

  // Register for storing intermediate data
  logic [31:0] data_reg;

  // Always block to update register and process array
  always_ff @(posedge clk) begin
    data_reg <= in_data;
    packet_array[0].payload <= data_reg;
    packet_array[0].header <= data_reg[7:0];
    
    // Process remaining array elements
    for (int i = 1; i < 4; i++) begin
      packet_array[i].payload <= packet_array[i-1].payload;
      packet_array[i].header <= packet_array[i-1].header;
    end
  end

  // Generate result from packet array
  always_comb begin
    result = packet_array[0].payload[3:0] ^ 
             packet_array[1].payload[3:0] ^ 
             packet_array[2].payload[3:0] ^ 
             packet_array[3].payload[3:0];
  end

endmodule