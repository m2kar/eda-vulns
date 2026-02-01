# Minimization Report

## Input
`source.sv`

## Output
`bug.sv`

## Strategy
Removed the parameter computation and reset/clock plumbing, keeping only the
combinational assignment and the assertion with `$error` that triggers
`sim.fmt.literal`.

## Reduction
* Lines: 12 → 10 (~16.7% reduction)
* Semantics preserved: compile-time legalization failure on `sim.fmt.literal`.
