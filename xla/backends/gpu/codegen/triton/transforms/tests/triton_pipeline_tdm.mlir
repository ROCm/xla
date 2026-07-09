// RUN: xla-opt %s -split-input-file --triton-xla-pipeline='target=gfx1250' \
// RUN:   | FileCheck %s --check-prefix=CHECK-TDM
//
// RUN: xla-opt %s -split-input-file --triton-xla-pipeline='target=gfx950' \
// RUN:   | FileCheck %s --check-prefix=CHECK-NOTDM

// Verifies that the full Triton XLA + AMD lowering pipeline emits TDM
// intrinsics on gfx1250 and pointer-arithmetic buffer ops on non-TDM arches.

func.func @lower_extract_insert(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract from %arg0
      as memref<256x256xbf16, #xtile.layout<[1, 0]>>
      [0, 0] [16, 64] [1, 1] : tensor<16x64xbf16>
  triton_xla.insert %extracted_tensor into %arg1
      as memref<256x256xbf16, #xtile.layout<[1, 0]>>
      [0, 0] [16, 64] [1, 1] : tensor<16x64xbf16>
  func.return
}

// CHECK-TDM-LABEL: llvm.func @lower_extract_insert
// CHECK-TDM:       tensor.load.to.lds
// CHECK-TDM:       s.wait.tensorcnt
// CHECK-TDM:       tensor.store.from.lds

// CHECK-NOTDM-LABEL: llvm.func @lower_extract_insert
// CHECK-NOTDM-NOT:   tensor.load.to.lds
// CHECK-NOTDM-NOT:   tensor.store.from.lds
// CHECK-NOTDM:       raw.ptr.buffer.load

// -----

// End-to-end batched-dot test with a trivial (tile size 1) batch dim on
// both operands, through the full pipeline including tt.dot. Regression
// test for the fold-into-base-pointer fix for TDM descriptors whose tile
// shape has a trivial trailing dim -- see
// triton_issue_padded_shared_order.md.
func.func @batched_dot(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>,
                       %arg2: !tt.ptr<bf16>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32>

  %lhs_tile = triton_xla.extract from %arg0
      as memref<32x256x4xbf16, #xtile.layout<[2, 1, 0]>>
      [0, 0, 2] [32, 256, 1] [1, 1, 1] : tensor<32x256x1xbf16>
  %lhs = tt.reshape %lhs_tile : tensor<32x256x1xbf16> -> tensor<32x256xbf16>

  %rhs_tile = triton_xla.extract from %arg1
      as memref<32x256x4xbf16, #xtile.layout<[2, 1, 0]>>
      [0, 0, 2] [32, 256, 1] [1, 1, 1] : tensor<32x256x1xbf16>
  %rhs_t = tt.trans %rhs_tile {order = array<i32: 1, 0, 2>}
      : tensor<32x256x1xbf16> -> tensor<256x32x1xbf16>
  %rhs = tt.reshape %rhs_t : tensor<256x32x1xbf16> -> tensor<256x32xbf16>

  %dot = tt.dot %lhs, %rhs, %cst, inputPrecision = tf32
      : tensor<32x256xbf16> * tensor<256x32xbf16> -> tensor<32x32xf32>
  %dot_bf16 = arith.truncf %dot : tensor<32x32xf32> to tensor<32x32xbf16>

  triton_xla.insert %dot_bf16 into %arg2
      as memref<32x32xbf16, #xtile.layout<[1, 0]>>
      [0, 0] [32, 32] [1, 1] : tensor<32x32xbf16>
  func.return
}

// CHECK-TDM-LABEL: llvm.func @batched_dot
// CHECK-TDM:       tensor.load.to.lds
// CHECK-TDM:       tensor.load.to.lds
// CHECK-TDM:       s.wait.tensorcnt
// CHECK-TDM:       tensor.store.from.lds

// CHECK-NOTDM-LABEL: llvm.func @batched_dot
// CHECK-NOTDM-NOT:   tensor.load.to.lds
// CHECK-NOTDM-NOT:   tensor.store.from.lds
// CHECK-NOTDM:       raw.ptr.buffer.load

// -----

// Genuine (non-trivial, tile size > 1) rank-3 tile, e.g. all 4 batches of a
// batched matmul operand handled per kernel instance. There is no trivial
// dim to fold here, so this exercises the ordinary rank-3 TDM path through
// the full pipeline, unaffected by the fold above -- included for
// contrast/coverage alongside @batched_dot. (Tile size 4 rather than an
// arbitrary >1 value like 2: TDM's padding interval must be at least 2
// dwords, and 2 bf16 elements in the minor-most dim is only 4 bytes.)
func.func @lower_extract_insert_3d(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
  %extracted_tensor = triton_xla.extract from %arg0
      as memref<32x256x4xbf16, #xtile.layout<[2, 1, 0]>>
      [0, 0, 0] [32, 256, 4] [1, 1, 1] : tensor<32x256x4xbf16>
  triton_xla.insert %extracted_tensor into %arg1
      as memref<32x256x4xbf16, #xtile.layout<[2, 1, 0]>>
      [0, 0, 0] [32, 256, 4] [1, 1, 1] : tensor<32x256x4xbf16>
  func.return
}

// CHECK-TDM-LABEL: llvm.func @lower_extract_insert_3d
// CHECK-TDM:       tensor.load.to.lds
// CHECK-TDM:       s.wait.tensorcnt
// CHECK-TDM:       tensor.store.from.lds

// CHECK-NOTDM-LABEL: llvm.func @lower_extract_insert_3d
// CHECK-NOTDM-NOT:   tensor.load.to.lds
// CHECK-NOTDM-NOT:   tensor.store.from.lds
// CHECK-NOTDM:       raw.ptr.buffer.load

// -----

// A more realistic batched-dot kernel: one grid program loops over all 4
// batches itself (rather than one grid program per batch, as in
// @batched_dot above), so the trivial-batch-dim tile's offset is a
// loop-carried dynamic value instead of a compile-time constant. Stresses
// the fold-into-base-pointer fix's tt.addptr codegen with a runtime offset.
// NOTE: unlike @batched_dot's grid-per-batch scheme, this recomputes
// tt.make_tensor_descriptor + tt.addptr on every loop iteration (the
// canonical Triton pattern instead hoists the descriptor out of the loop
// and only threads a per-iteration offset into tt.descriptor_load, via an
// scf.for iter_arg -- see the doc comment on
// TritonAMDGPUOptimizeDescriptorEncodingPass). Recomputing per iteration is
// still correct, just not necessarily the most efficient lowering.
func.func @batched_dot_loop(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>,
                            %arg2: !tt.ptr<bf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %b = %c0 to %c4 step %c1 {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32>

    %lhs_tile = triton_xla.extract from %arg0
        as memref<32x256x4xbf16, #xtile.layout<[2, 1, 0]>>
        [0, 0, %b] [32, 256, 1] [1, 1, 1] : tensor<32x256x1xbf16>
    %lhs = tt.reshape %lhs_tile : tensor<32x256x1xbf16> -> tensor<32x256xbf16>

    %rhs_tile = triton_xla.extract from %arg1
        as memref<32x256x4xbf16, #xtile.layout<[2, 1, 0]>>
        [0, 0, %b] [32, 256, 1] [1, 1, 1] : tensor<32x256x1xbf16>
    %rhs_t = tt.trans %rhs_tile {order = array<i32: 1, 0, 2>}
        : tensor<32x256x1xbf16> -> tensor<256x32x1xbf16>
    %rhs = tt.reshape %rhs_t : tensor<256x32x1xbf16> -> tensor<256x32xbf16>

    %dot = tt.dot %lhs, %rhs, %cst, inputPrecision = tf32
        : tensor<32x256xbf16> * tensor<256x32xbf16> -> tensor<32x32xf32>
    %dot_bf16 = arith.truncf %dot : tensor<32x32xf32> to tensor<32x32xbf16>

    triton_xla.insert %dot_bf16 into %arg2
        as memref<32x32x4xbf16, #xtile.layout<[2, 1, 0]>>
        [0, 0, %b] [32, 32, 1] [1, 1, 1] : tensor<32x32xbf16>
    scf.yield
  }
  func.return
}

// CHECK-TDM-LABEL: llvm.func @batched_dot_loop
// CHECK-TDM:       tensor.load.to.lds
// CHECK-TDM:       tensor.load.to.lds
// CHECK-TDM:       s.wait.tensorcnt
// CHECK-TDM:       tensor.store.from.lds

// CHECK-NOTDM-LABEL: llvm.func @batched_dot_loop
// CHECK-NOTDM-NOT:   tensor.load.to.lds
// CHECK-NOTDM-NOT:   tensor.store.from.lds
// CHECK-NOTDM:       raw.ptr.buffer.load
