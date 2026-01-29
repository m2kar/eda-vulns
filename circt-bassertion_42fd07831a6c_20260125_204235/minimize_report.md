# Minimize Report

## Summary
- **Original Size**: 12 lines (284 chars)
- **Minimized Size**: 2 lines (39 chars)
- **Reduction**: 83.3% (from 12 to 2 lines)

## Minimization Strategy
Based on root cause analysis, the crash is triggered by **string type port declaration**.

### Key Construct Preserved
- `output string str` → string type as module port

### Removed Constructs
1. `input logic in` - unnecessary port
2. `output logic out` - unnecessary port  
3. `logic x` - unnecessary variable
4. `always_comb` block - unrelated logic
5. `initial` block - unrelated logic

## Original Test Case
```systemverilog
module test(input logic in, output logic out, output string str);
  logic x;
  
  always_comb begin
    x = in;
    out = x;
  end
  
  initial begin
    str = "Hello";
  end
endmodule
```

## Minimized Test Case
```systemverilog
module test(output string str);
endmodule
```

## Verification
- ✅ Minimized test case reproduces the same assertion failure
- ✅ Same crash signature: `SVModuleOpConversion::matchAndRewrite MooreToCore.cpp`
- ✅ Same assertion: `detail::isPresent(Val) && "dyn_cast on a non-existent value"`

## Command
```bash
circt-verilog --ir-hw bug.sv
```

## Notes
- The crash is triggered solely by having a `string` type port
- Both `input string` and `output string` trigger the crash
- The module body content is irrelevant to the bug
