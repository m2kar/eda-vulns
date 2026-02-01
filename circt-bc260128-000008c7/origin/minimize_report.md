# Minimization Report

## Input
* Original testcase: `source.sv`
* Goal: preserve arc/state type mismatch triggering arcilator failure

## Result
No further reduction found beyond the existing minimal module. The testcase is already a
small, single-module design exercising a packed array of structs with a reset loop and a
simple read.

## Output Files
* `bug.sv` (identical to `source.sv`)
* `error.log`
* `command.txt`

## Reduction
* Lines reduced: 0
* Reduction percent: 0%
