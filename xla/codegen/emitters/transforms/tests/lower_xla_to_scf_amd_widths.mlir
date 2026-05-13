// RUN: emitters_opt %s --split-input-file \
// RUN:     -xla-lower-xla-to-scf="warp_size=64 use_dpp_subgroup_widths=true" \
// RUN: | FileCheck %s

// With use_dpp_subgroup_widths the producer emits gpu.shuffle widths that
// match the AMDGPU DPP / ds_swizzle granularity:
//   - distances in [1, 15] -> width 16 (DPP row)
//   - distance == 16       -> width 32 (ds_swizzle group)
//   - larger distances     -> warp_size

func.func @combiner(%a: f32, %b: f32) -> f32 {
  return %a : f32
}

func.func @shuffler_to_32(%a: f32) -> f32 {
  %ret = xla_gpu.shuffle_reduce(%a) to 32 combiner=@combiner : f32
  return %ret : f32
}
// CHECK-LABEL: @shuffler_to_32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK-DAG: %[[C2:.*]] = arith.constant 2
// CHECK-DAG: %[[C4:.*]] = arith.constant 4
// CHECK-DAG: %[[C8:.*]] = arith.constant 8
// CHECK-DAG: %[[C16:.*]] = arith.constant 16
// CHECK-DAG: %[[C32:.*]] = arith.constant 32
// CHECK-DAG: %[[C64:.*]] = arith.constant 64
// CHECK: gpu.shuffle down {{.*}}, %[[C32]], %[[C64]]
// CHECK: gpu.shuffle down {{.*}}, %[[C16]], %[[C32]]
// CHECK: gpu.shuffle down {{.*}}, %[[C8]], %[[C16]]
// CHECK: gpu.shuffle down {{.*}}, %[[C4]], %[[C16]]
// CHECK: gpu.shuffle down {{.*}}, %[[C2]], %[[C16]]
// CHECK: gpu.shuffle down {{.*}}, %[[C1]], %[[C16]]

// -----

func.func @combiner(%a: f32, %b: f32) -> f32 {
  return %a : f32
}

func.func @shuffler_to_8(%a: f32) -> f32 {
  %ret = xla_gpu.shuffle_reduce(%a) to 8 combiner=@combiner : f32
  return %ret : f32
}
// All distances are < 16, so all shuffles should use width 16.
// CHECK-LABEL: @shuffler_to_8
// CHECK-DAG: %[[C16:.*]] = arith.constant 16
// CHECK-COUNT-4: gpu.shuffle down {{.*}}, {{.*}}, %[[C16]]
// CHECK-NOT: gpu.shuffle down
