module test_module (
    input logic signed [7:0] signed_port,
    input logic [7:0] unsigned_port,
    output logic [7:0] out_port,
    inout logic io_port
);

    logic [7:0] reg_data;

    always_comb begin
        if (unsigned_port > signed_port) // Type difference: unsigned vs signed comparison
            reg_data = unsigned_port;
        else
            reg_data = signed_port;
    end

    assign out_port = reg_data;
    assign io_port = (out_port[0]) ? 1'bz : 1'b0;

endmodule