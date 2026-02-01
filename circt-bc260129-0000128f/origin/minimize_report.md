# Minimization Report

## Outcome
No further reduction was possible without losing the Arc/HW state construction
that triggers the type mismatch. The original testcase is already compact and
focused, so the minimized testcase is identical to the input.

## Notes
- Key constructs retained:
  - Variable-bound loop writing an array of registers.
  - Subsequent read from the array to drive output.
- These constructs appear necessary to produce the Arc `state` operation and
  the related type mismatch observed during arcilator processing.
