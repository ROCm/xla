// RUN: xla-opt %s -triton-xla-atomics-rocm | FileCheck %s

// Test lowering of AtomicWriteOp to pure Triton AtomicRMWOp for ROCm

// CHECK-LABEL: @test_atomic_write
tt.func @test_atomic_write(%ptr: !tt.ptr<i32>, %value: i32) {
  // CHECK-NOT: triton_xla.atomic_write
  // CHECK: tt.atomic_rmw xchg
  triton_xla.atomic_write sys, release, %ptr, %value : (!tt.ptr<i32>, i32) -> ()
  tt.return
}

// CHECK-LABEL: @test_atomic_write_with_mask
tt.func @test_atomic_write_with_mask(%ptr: !tt.ptr<i32>, %value: i32, %mask: i1) {
  // CHECK-NOT: triton_xla.atomic_write
  // CHECK: tt.atomic_rmw xchg
  triton_xla.atomic_write sys, release, %ptr, %value, %mask : (!tt.ptr<i32>, i32, i1) -> ()
  tt.return
}

// CHECK-LABEL: @test_atomic_spin_wait
tt.func @test_atomic_spin_wait(%ptr: !tt.ptr<i32>, %expected: i32) {
  // CHECK-NOT: triton_xla.atomic_spin_wait
  // CHECK: scf.while
  // CHECK: tt.atomic_cas
  // CHECK: arith.cmpi
  // CHECK: scf.condition
  triton_xla.atomic_spin_wait sys, acquire, %ptr, lt, %expected : (!tt.ptr<i32>, i32) -> ()
  tt.return
}

// CHECK-LABEL: @test_atomic_spin_wait_eq
tt.func @test_atomic_spin_wait_eq(%ptr: !tt.ptr<i32>, %expected: i32) {
  // CHECK-NOT: triton_xla.atomic_spin_wait
  // CHECK: scf.while
  // CHECK: tt.atomic_cas
  // CHECK: arith.cmpi eq
  triton_xla.atomic_spin_wait sys, acquire, %ptr, eq, %expected : (!tt.ptr<i32>, i32) -> ()
  tt.return
}

// CHECK-LABEL: @test_atomic_operations_tensor
tt.func @test_atomic_operations_tensor(%ptr: tensor<4x!tt.ptr<i32>>, %value: tensor<4xi32>) {
  // CHECK-NOT: triton_xla.atomic_write
  // CHECK: tt.atomic_rmw xchg
  triton_xla.atomic_write sys, release, %ptr, %value : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>) -> ()
  tt.return
}
