; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare void @exit(i32)

define void @MixPorts_eval(ptr %0) {
  %2 = load i64, ptr %0, align 4
  %3 = trunc i64 %2 to i32
  %4 = getelementptr i8, ptr %0, i32 12
  store i32 %3, ptr %4, align 4
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
