module test_module(output string str_out, output logic [7:0] count_out);
  string a = "Test";
  logic [7:0] count;

  // Assign outputs
  assign str_out = a;
  assign count_out = count;

  // Trigger logic on string length change
  always @(a.len()) begin
    count <= 8'd0;
  end
endmodule