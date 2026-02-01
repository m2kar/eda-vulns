# Minimization Report

## Input
- Original testcase: source.sv

## Result
- Reduced testcase: bug.sv
- Reduction: 0% (original already minimal for reproducing crash)

## Notes
The testcase is already a minimal module exercising an SV `string` input and
`string.len()`; removing any statement or port eliminates the trigger.
