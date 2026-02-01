typedef enum logic [1:0] { STATE_A, STATE_B, STATE_C } state_t;
typedef struct packed { state_t state; logic [3:0] data; } state_data_t;

module submod(input logic [3:0] data_in);
  // Submodule implementation (placeholder)
endmodule

module top(output state_data_t out, output logic [3:0] shared_data);
  
  always_comb begin
    case (out.state)
      STATE_A: out.data = 4'h1;
      STATE_B: out.data = 4'h2;
      STATE_C: out.data = 4'h3;
      default: out.data = 4'h0;
    endcase
  end
  
  assign shared_data = out.data;
  
  submod inst1 (.data_in(shared_data));
  
endmodule