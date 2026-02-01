# [MooreToCore] Emit targeted diagnostics for unsupported module port types (fixes #9572)

## Overview
Improve diagnostics and failure handling for unsupported port types in the Moore-to-Core conversion. Previously, converting a `moore.module` to `hw.module` could proceed with invalid port information when a port type failed to convert, leading to a generic legalization failure with no clear root cause. This change validates port-type conversion during port-info construction and emits a precise error when an unsupported type is encountered.

Fixes issue #9572

## Background
In the `MooreToCore` conversion, `SVModuleOp` port types are converted through the `TypeConverter`. When the conversion returns a null type, the existing code still builds `ModulePortInfo`, which later results in “failed to legalize” without actionable diagnostics.

## Changes
- Change `getModulePortInfo` to return `FailureOr<hw::ModulePortInfo>`.
- Emit a targeted error on the original `moore.module` when a port type cannot be converted, including the port name and original type.
- Propagate `failure()` in the `SVModuleOp` conversion to prevent creating an invalid `hw.module`.

## Behavior
- Modules with unsupported port types now produce a precise diagnostic:
  - `port '<name>' has unsupported type '<type>' that cannot be converted to hardware type`
- The conversion fails early while keeping IR consistent.

## Tests
Updated [test/Conversion/MooreToCore/errors.mlir](test/Conversion/MooreToCore/errors.mlir) to cover:
- A single unsupported input port type.
- Mixed ports where only one port is unsupported.
- Retaining the `failed to legalize` expectation while surfacing the more specific error first.

## Scope
- Limited to port-type validation and diagnostics in the `MooreToCore` conversion.
- No change in behavior for convertible types.

## References
- Fix in [lib/Conversion/MooreToCore/MooreToCore.cpp](lib/Conversion/MooreToCore/MooreToCore.cpp)
- Tests in [test/Conversion/MooreToCore/errors.mlir](test/Conversion/MooreToCore/errors.mlir)
