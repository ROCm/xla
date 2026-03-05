// RUN: xla-opt %s -triton-xla-get-tid-rocm | FileCheck %s

// Test lowering of GetTidOp to pure Triton GetThreadIdOp for ROCm

// CHECK-LABEL: @test_get_tid
tt.func @test_get_tid() -> i32 {
  // CHECK-NOT: triton_xla.get_tid
  // CHECK: tt.get_thread_id
  %tid = triton_xla.get_tid : () -> i32
  tt.return %tid : i32
}

// CHECK-LABEL: @test_get_tid_usage
tt.func @test_get_tid_usage(%world_size: i32) -> i1 {
  // CHECK-NOT: triton_xla.get_tid
  // CHECK: [[TID:%.*]] = tt.get_thread_id
  // CHECK: arith.cmpi ult, [[TID]], %world_size
  %tid = triton_xla.get_tid : () -> i32
  %cond = arith.cmpi ult, %tid, %world_size : i32
  tt.return %cond : i1
}
