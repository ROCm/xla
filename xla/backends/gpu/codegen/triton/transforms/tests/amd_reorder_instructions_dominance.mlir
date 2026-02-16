// RUN: xla-opt %s -tritonamdgpu-reorder-instructions 2>&1 | FileCheck %s

// Verify that reorder-instructions does not produce dominance violations
// when load results are used inside nested regions (scf.if within scf.for).

// CHECK-NOT: does not dominate this use
// CHECK-LABEL: @sink_second_load_nested_if
// CHECK:       scf.for
// CHECK:         tt.load
// CHECK:         tt.load
// CHECK:         scf.if
// CHECK-NOT:     tt.load
// CHECK:         tt.dot
#blk = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 2], instrShape = [32, 32, 16], isTransposed = true}>
#sh = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @sink_second_load_nested_if(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %cond: i1) {
    %zero = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %pad_A = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blk>
    %pad_B = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #blk>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %ptrA = tt.splat %A : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blk>
    %ptrB = tt.splat %B : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blk>
    %r = scf.for %iv = %c0 to %c2 step %c1 iter_args(%acc = %zero) -> (tensor<128x128xf32, #mma>) {
      %ldA = tt.load %ptrA : tensor<128x64x!tt.ptr<f16>, #blk>
      %ldB = tt.load %ptrB : tensor<64x128x!tt.ptr<f16>, #blk>
      %sel:2 = scf.if %cond -> (tensor<128x64xf16, #blk>, tensor<64x128xf16, #blk>) {
        scf.yield %ldA, %ldB : tensor<128x64xf16, #blk>, tensor<64x128xf16, #blk>
      } else {
        scf.yield %pad_A, %pad_B : tensor<128x64xf16, #blk>, tensor<64x128xf16, #blk>
      }
      %sA = ttg.local_alloc %sel#0 : (tensor<128x64xf16, #blk>) -> !ttg.memdesc<128x64xf16, #sh, #smem>
      %dA = ttg.local_load %sA : !ttg.memdesc<128x64xf16, #sh, #smem> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %sB = ttg.local_alloc %sel#1 : (tensor<64x128xf16, #blk>) -> !ttg.memdesc<64x128xf16, #sh, #smem>
      %dB = ttg.local_load %sB : !ttg.memdesc<64x128xf16, #sh, #smem> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %dot = tt.dot %dA, %dB, %acc, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      scf.yield %dot : tensor<128x128xf32, #mma>
    }
    tt.return
  }
}
