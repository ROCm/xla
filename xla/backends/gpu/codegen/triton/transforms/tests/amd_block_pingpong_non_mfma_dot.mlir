// RUN: xla-opt %s --triton-xla-pipeline='target=gfx942' | FileCheck %s

// Verify that the block-pingpong optimization does not crash on non-MFMA
// (FMA) dot operations. The small tile sizes here (16x16 * 16x8) cause
// Triton to lower to FMA instead of MFMA instructions.

// CHECK: llvm.func @kernel
module {
  tt.func @kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<16x8xf32>
    %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x8x!tt.ptr<f32>>
    %2 = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%arg4 = %cst) -> (tensor<16x8xf32>) {
      %3 = tt.load %0 : tensor<16x16x!tt.ptr<f32>>
      %4 = tt.load %1 : tensor<16x8x!tt.ptr<f32>>
      %5 = tt.dot %3, %4, %arg4, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x8xf32> -> tensor<16x8xf32>
      scf.yield %5 : tensor<16x8xf32>
    }
    %6 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x8x!tt.ptr<f32>>
    tt.store %6, %2 : tensor<16x8x!tt.ptr<f32>>
    tt.return
  }
}
