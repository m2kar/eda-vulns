module test_module(
    input real in_real,
    input logic clk,
    output real out_real,
    output logic cmp_result
);

    wire signed [1:0] a = 2'b10;
    
    always_comb begin
        cmp_result = (-a <= a) ? 1 : 0;
    end
    
    always @(posedge clk) begin
        out_real <= in_real;
    end

endmodule