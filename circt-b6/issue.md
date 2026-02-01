# [Arc] Assertion failure when lowering inout ports in sequential logic

## Description

CIRCT crashes with an assertion failure when compiling SystemVerilog code that uses `inout` ports within `always_ff` blocks. The crash occurs in the Arc dialect's `LowerStatePass` when attempting to create a `StateType` for an LLHD reference type.

## Steps to Reproduce

1. Save the following code as `bug.sv`:

```systemverilog
module MixedPorts(
  inout wire c,
  input logic clk
);
  logic temp_reg;

  always_ff @(posedge clk) begin
    temp_reg <= c;
  end
endmodule
```

2. Run:
```bash
circt-verilog --ir-hw bug.sv | arcilator
```

## Expected Behavior

Either successful compilation or a graceful error message such as:
```
error: inout ports cannot be used directly in sequential logic
```

## Actual Behavior

Compiler crashes with assertion failure:

```
<unknown>:0: error: state type must have a known bit width; got '!llhd.ref<i1>'
arcilator: /root/circt/llvm/llvm/../mlir/include/mlir/IR/StorageUniquerSupport.h:180: static ConcreteT mlir::detail::StorageUserBase<circt::arc::StateType, mlir::Type, circt::arc::detail::StateTypeStorage, mlir::detail::TypeUniquer>::get(MLIRContext *, Args &&...) [ConcreteT = circt::arc::StateType, BaseT = mlir::Type, StorageT = circt::arc::detail::StateTypeStorage, UniquerT = mlir::detail::TypeUniquer, Traits = <>, Args = <mlir::Type &>]: Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace and instructions to reproduce the bug.
Stack dump:
0.      Program arguments: arcilator
 #0 0x0000559d10553d88 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) /root/circt/llvm/llvm/lib/Support/Unix/Signals.inc:842:13
 #1 0x0000559d10551873 llvm::sys::RunSignalHandlers() /root/circt/llvm/llvm/lib/Support/Signals.cpp:109:18
 #2 0x0000559d10554d61 SignalHandler(int, siginfo_t*, void*) /root/circt/llvm/llvm/lib/Support/Unix/Signals.inc:429:38
 #3 0x00007f400f7c6330 (/lib/x86_64-linux-gnu/libc.so.6+0x45330)
 #4 0x00007f400f81fb2c __pthread_kill_implementation ./nptl/pthread_kill.c:44:76
 #5 0x00007f400f81fb2c __pthread_kill_internal ./nptl/pthread_kill.c:78:10
 #6 0x00007f400f81fb2c pthread_kill ./nptl/pthread_kill.c:89:10
 #7 0x00007f400f7c627e raise ./signal/../sysdeps/posix/raise.c:27:6
 #8 0x00007f400f7a98ff abort ./stdlib/abort.c:81:7
 #9 0x00007f400f7a981b _nl_load_domain ./intl/loadmsgcat.c:1177:9
#10 0x00007f400f7bc517 (/lib/x86_64-linux-gnu/libc.so.6+0x3b517)
#11 0x0000559d1098a9a7 (/home/zhiqing/edazz/circt/build/bin/arcilator+0x6b69a7)
#12 0x0000559d1098a8e2 circt::arc::StateType::get(mlir::Type) /root/circt/build/tools/circt/include/circt/Dialect/Arc/ArcTypes.cpp.inc:108:3
#13 0x0000559d109f8768 (anonymous namespace)::ModuleLowering::run() /root/circt/lib/Dialect/Arc/Transforms/LowerState.cpp:219:66
#14 0x0000559d109f8768 (anonymous namespace)::LowerStatePass::runOnOperation() /root/circt/lib/Dialect/Arc/Transforms/LowerState.cpp:1198:41
#15 0x0000559d12baad82 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)::$_3::operator()() const /root/circt/llvm/mlir/lib/Pass/Pass.cpp:0:19
#16 0x0000559d12baad82 void llvm::function_ref<void ()>::callback_fn<mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int)::$_3>(long) /root/circt/llvm/llvm/include/llvm/ADT/STLFunctionalExtras.h:46:12
#17 0x0000559d12baad82 llvm::function_ref<void ()>::operator()() const /root/circt/llvm/llvm/include/llvm/ADT/STLFunctionalExtras.h:69:12
#18 0x0000559d12baad82 void mlir::MLIRContext::executeAction<mlir::PassExecutionAction, mlir::Pass&>(llvm::function_ref<void ()>, llvm::ArrayRef<mlir::IRUnit>, mlir::Pass&) /root/circt/llvm/mlir/include/mlir/IR/MLIRContext.h:290:7
#19 0x0000559d12baad82 mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) /root/circt/llvm/mlir/lib/Pass/Pass.cpp:606:23
#20 0x0000559d12baba44 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) /root/circt/llvm/mlir/lib/Pass/Pass.cpp:688:9
#21 0x0000559d12bb268b mlir::PassManager::runPasses(mlir::Operation*, mlir::AnalysisManager) /root/circt/llvm/mlir/lib/Pass/Pass.cpp:1123:3
#22 0x0000559d12bb1d0e mlir::PassManager::run(mlir::Operation*) /root/circt/llvm/mlir/lib/Pass/Pass.cpp:1097:0
#23 0x0000559d104e2f25 processBuffer(mlir::MLIRContext&, mlir::TimingScope&, llvm::SourceMgr&, std::optional<std::unique_ptr<llvm::ToolOutputFile, std::default_delete<llvm::ToolOutputFile>>>&) /root/circt/tools/arcilator/arcilator.cpp:404:7
#24 0x0000559d104e209f processInputSplit(mlir::MLIRContext&, mlir::TimingScope&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, std::optional<std::unique_ptr<llvm::ToolOutputFile, std::default_delete<llvm::ToolOutputFile>>>&) /root/circt/tools/arcilator/arcilator.cpp:637:12
#25 0x0000559d104df57a processInput(mlir::MLIRContext&, mlir::TimingScope&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, std::optional<std::unique_ptr<llvm::ToolOutputFile, std::default_delete<llvm::ToolOutputFile>>>&) /root/circt/tools/arcilator/arcilator.cpp:653:12
#26 0x0000559d104df57a executeArcilator(mlir::MLIRContext&) /root/circt/tools/arcilator/arcilator.cpp:720:14
#27 0x0000559d104dec1d llvm::LogicalResult::failed() const /root/circt/llvm/llvm/include/llvm/Support/LogicalResult.h:43:42
#28 0x0000559d104dec1d llvm::failed(llvm::LogicalResult) /root/circt/llvm/llvm/include/llvm/Support/LogicalResult.h:71:58
#29 0x0000559d104dec1d main /root/circt/tools/arcilator/arcilator.cpp:788:8
#30 0x00007f400f7ab1ca __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:74:3
#31 0x00007f400f7ab28b call_init ./csu/../csu/libc-start.c:128:20
#32 0x00007f400f7ab28b __libc_start_main ./csu/../csu/libc-start.c:347:5
#33 0x0000559d104ddfd5 _start (/home/zhiqing/edazz/circt/build/bin/arcilator+0x209fd5)
command.txt: line 1: 1314381 Done                    circt-verilog --ir-hw bug.sv
     1314382 Aborted                 (core dumped) | arcilator
```

## Root Cause Analysis

The issue occurs in the following flow:

1. **Frontend (circt-verilog)**: Parses the inout port and represents it as `!llhd.ref<i1>` (LLHD reference type)
2. **Arc LowerStatePass**: Attempts to create state storage for the `always_ff` block that reads from the inout port
3. **Type Verification Failure**: `StateType::get()` calls `verifyInvariants()`, which requires types with known bit widths
4. **Crash**: LLHD reference types are opaque pointers without intrinsic bit width, causing the assertion to fail

**Crash Location**: `lib/Dialect/Arc/Transforms/LowerState.cpp:219` in `ModuleLowering::run()`

## Environment

- **CIRCT Version**: commit e4838c703  (BTW: v1.139.0 can't reproduce)
- **LLVM Version**: 23.0.0git
- **Build Type**: Optimized build with assertions
- **Affected Tool**: arcilator
- **Affected Pass**: LowerStatePass
