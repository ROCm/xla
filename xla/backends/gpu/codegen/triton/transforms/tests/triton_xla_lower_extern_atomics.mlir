// RUN: xla-opt %s -triton-xla-lower-extern-atomics | FileCheck %s

// Test lowering of AtomicWriteOp and AtomicSpinWaitOp to tt.extern_elementwise

// CHECK-LABEL: @test_atomic_write
tt.func @test_atomic_write(%ptr: !tt.ptr<i32>, %value: i32) {
  // CHECK-NOT: triton_xla.atomic_write
  // CHECK: tt.extern_elementwise
  // CHECK-SAME: symbol = "xla_atomic_write_release_system"
  triton_xla.atomic_write sys, release, %ptr, %value : (!tt.ptr<i32>, i32) -> ()
  tt.return
}

// CHECK-LABEL: @test_atomic_write_vectorized
tt.func @test_atomic_write_vectorized(%ptr: tensor<4x!tt.ptr<i32>>, %value: i32) {
  // CHECK-NOT: triton_xla.atomic_write
  // CHECK: [[SPLAT:%.*]] = tt.splat %arg1 : i32 -> tensor<4xi32>
  // CHECK: tt.extern_elementwise {{%.*}}, [[SPLAT]]
  // CHECK-SAME: symbol = "xla_atomic_write_release_system"
  triton_xla.atomic_write sys, release, %ptr, %value : (tensor<4x!tt.ptr<i32>>, i32) -> ()
  tt.return
}

// CHECK-LABEL: @test_atomic_spin_wait_vectorized
tt.func @test_atomic_spin_wait_vectorized(%ptr: tensor<4x!tt.ptr<i32>>, %expected: i32) {
  // CHECK-NOT: triton_xla.atomic_spin_wait
  // CHECK: [[SPLAT:%.*]] = tt.splat %arg1 : i32 -> tensor<4xi32>
  // CHECK: tt.extern_elementwise {{%.*}}, [[SPLAT]]
  // CHECK-SAME: symbol = "xla_atomic_spin_wait_acquire_system_lt"
  triton_xla.atomic_spin_wait sys, acquire, %ptr, less_than, %expected : (tensor<4x!tt.ptr<i32>>, i32) -> ()
  tt.return
}

// CHECK-LABEL: @test_atomic_spin_wait_eq_vectorized
tt.func @test_atomic_spin_wait_eq_vectorized(%ptr: tensor<4x!tt.ptr<i32>>, %expected: i32) {
  // CHECK-NOT: triton_xla.atomic_spin_wait
  // CHECK: [[SPLAT:%.*]] = tt.splat %arg1 : i32 -> tensor<4xi32>
  // CHECK: tt.extern_elementwise {{%.*}}, [[SPLAT]]
  // CHECK-SAME: symbol = "xla_atomic_spin_wait_acquire_system_eq"
  triton_xla.atomic_spin_wait sys, acquire, %ptr, equal_to, %expected : (tensor<4x!tt.ptr<i32>>, i32) -> ()
  tt.return
}

// CHECK-LABEL: @test_different_scopes
tt.func @test_different_scopes(%ptr: tensor<4x!tt.ptr<i32>>, %value: i32) {
  // CHECK: tt.extern_elementwise
  // CHECK-SAME: symbol = "xla_atomic_write_release_gpu"
  triton_xla.atomic_write gpu, release, %ptr, %value : (tensor<4x!tt.ptr<i32>>, i32) -> ()
  
  // CHECK: tt.extern_elementwise
  // CHECK-SAME: symbol = "xla_atomic_write_release_cta"
  triton_xla.atomic_write cta, release, %ptr, %value : (tensor<4x!tt.ptr<i32>>, i32) -> ()
  
  tt.return
}
