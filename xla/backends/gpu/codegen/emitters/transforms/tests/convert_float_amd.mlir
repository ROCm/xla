// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: emitters_opt %s -split-input-file -xla-gpu-convert-float-amd="gpu_device_info='rocm_compute_capability {gcn_arch_name: \"gfx942:sramecc+:xnack\"}'" -canonicalize | FileCheck %s
// RUN: emitters_opt %s -split-input-file -xla-gpu-convert-float-amd="gpu_device_info='rocm_compute_capability {gcn_arch_name: \"gfx950:sramecc+:xnack-\"}'" -canonicalize | FileCheck %s --check-prefix=GFX950

module {
  func.func @intr_f16_to_f8(%arg0: f16) -> (f8E4M3FNUZ, f8E5M2FNUZ) {
    %a = arith.truncf %arg0 : f16 to f8E4M3FNUZ
    %b = arith.truncf %arg0 : f16 to f8E5M2FNUZ
    return %a, %b : f8E4M3FNUZ, f8E5M2FNUZ
  }
}

// CHECK-LABEL: @intr_f16_to_f8
// CHECK: arith.extf %{{.+}} : f16 to f32
// CHECK: llvm.amdgcn.cvt.pk.fp8.f32
// CHECK: llvm.amdgcn.cvt.pk.bf8.f32

// -----

module {
  func.func @intr_bf16_to_f8(%arg0: bf16) -> (f8E4M3FNUZ, f8E5M2FNUZ) {
    %a = arith.truncf %arg0 : bf16 to f8E4M3FNUZ
    %b = arith.truncf %arg0 : bf16 to f8E5M2FNUZ
    return %a, %b : f8E4M3FNUZ, f8E5M2FNUZ
  }
}

// CHECK-LABEL: @intr_bf16_to_f8
// CHECK: arith.extf %{{.+}} : bf16 to f32
// CHECK: llvm.amdgcn.cvt.pk.fp8.f32
// CHECK: llvm.amdgcn.cvt.pk.bf8.f32

// -----

module {
  func.func @intr_f32_to_f8(%arg0: f32) -> (f8E4M3FNUZ, f8E5M2FNUZ) {
    %a = arith.truncf %arg0 : f32 to f8E4M3FNUZ
    %b = arith.truncf %arg0 : f32 to f8E5M2FNUZ
    return %a, %b : f8E4M3FNUZ, f8E5M2FNUZ
  }
}

// CHECK-LABEL: @intr_f32_to_f8
// CHECK: llvm.amdgcn.cvt.pk.fp8.f32
// CHECK: llvm.amdgcn.cvt.pk.bf8.f32

// -----

module {
  func.func @intr_f64_to_f8(%arg0: f64) -> (f8E4M3FNUZ, f8E5M2FNUZ) {
    %a = arith.truncf %arg0 : f64 to f8E4M3FNUZ
    %b = arith.truncf %arg0 : f64 to f8E5M2FNUZ
    return %a, %b : f8E4M3FNUZ, f8E5M2FNUZ
  }
}

// CHECK-LABEL: @intr_f64_to_f8
// CHECK: arith.truncf %{{.+}} : f64 to f32
// CHECK: llvm.amdgcn.cvt.pk.fp8.f32
// CHECK: arith.truncf %{{.+}} : f64 to f32
// CHECK: llvm.amdgcn.cvt.pk.bf8.f32

// -----

module {
  func.func @intr_f8_to_f16(%arg0: f8E4M3FNUZ, %arg1: f8E5M2FNUZ) -> (f16, f16) {
    %a = arith.extf %arg0 : f8E4M3FNUZ to f16
    %b = arith.extf %arg1 : f8E5M2FNUZ to f16
    return %a, %b : f16, f16
  }
}

// CHECK-LABEL: @intr_f8_to_f16
// CHECK: llvm.amdgcn.cvt.f32.fp8
// CHECK: llvm.amdgcn.cvt.f32.bf8
// CHECK: arith.truncf %{{.+}} : f32 to f16

// -----

module {
  func.func @intr_f8_to_bf16(%arg0: f8E4M3FNUZ, %arg1: f8E5M2FNUZ) -> (bf16, bf16) {
    %a = arith.extf %arg0 : f8E4M3FNUZ to bf16
    %b = arith.extf %arg1 : f8E5M2FNUZ to bf16
    return %a, %b : bf16, bf16
  }
}

// CHECK-LABEL: @intr_f8_to_bf16
// CHECK: llvm.amdgcn.cvt.f32.fp8
// CHECK: llvm.amdgcn.cvt.f32.bf8
// CHECK: llvm.bitcast %{{.+}} : f32 to i32
// CHECK: llvm.bitcast %{{.+}} : i16 to bf16
// CHECK-NOT: arith.truncf %{{.+}} : f32 to bf16

// -----

module {
  func.func @intr_f8_to_f32(%arg0: f8E4M3FNUZ, %arg1: f8E5M2FNUZ) -> (f32, f32) {
    %a = arith.extf %arg0 : f8E4M3FNUZ to f32
    %b = arith.extf %arg1 : f8E5M2FNUZ to f32
    return %a, %b : f32, f32
  }
}

// CHECK-LABEL: @intr_f8_to_f32
// CHECK: llvm.amdgcn.cvt.f32.fp8
// CHECK: llvm.amdgcn.cvt.f32.bf8


// -----

module {
  func.func @intr_f8_to_f64(%arg0: f8E4M3FNUZ, %arg1: f8E5M2FNUZ) -> (f64, f64) {
    %a = arith.extf %arg0 : f8E4M3FNUZ to f64
    %b = arith.extf %arg1 : f8E5M2FNUZ to f64
    return %a, %b : f64, f64
  }
}

// CHECK-LABEL: @intr_f8_to_f64
// CHECK: llvm.amdgcn.cvt.f32.fp8
// CHECK: arith.extf %{{.+}} : f32 to f64
// CHECK: llvm.amdgcn.cvt.f32.bf8
// CHECK: arith.extf %{{.+}} : f32 to f64

// -----

module {
  func.func @intr_f16_to_4f8(%arg0: f16, %arg1: f16, %arg2: f16, %arg3: f16) -> (vector<4xf8E4M3FNUZ>, vector<4xf8E5M2FNUZ>) {
    %a0 = arith.truncf %arg0 : f16 to f8E4M3FNUZ
    %a1 = arith.truncf %arg1 : f16 to f8E4M3FNUZ
    %a2 = arith.truncf %arg2 : f16 to f8E4M3FNUZ
    %a3 = arith.truncf %arg3 : f16 to f8E4M3FNUZ
    %b0 = arith.truncf %arg0 : f16 to f8E5M2FNUZ
    %b1 = arith.truncf %arg1 : f16 to f8E5M2FNUZ
    %b2 = arith.truncf %arg2 : f16 to f8E5M2FNUZ
    %b3 = arith.truncf %arg3 : f16 to f8E5M2FNUZ
    %a_init = arith.constant dense<0.000000e+00> : vector<4xf8E4M3FNUZ>
    %b_init = arith.constant dense<0.000000e+00> : vector<4xf8E5M2FNUZ>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = vector.insert %a0, %a_init [%c0] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %1 = vector.insert %a1, %0 [%c1] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %2 = vector.insert %a2, %1 [%c2] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %a = vector.insert %a3, %2 [%c3] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %3 = vector.insert %b0, %b_init [%c0] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %4 = vector.insert %b1, %3 [%c1] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %5 = vector.insert %b2, %4 [%c2] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %b = vector.insert %b3, %5 [%c3] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    return %a, %b : vector<4xf8E4M3FNUZ>, vector<4xf8E5M2FNUZ>
  }
}

// CHECK-LABEL: @intr_f16_to_4f8
// CHECK-COUNT-4: arith.extf %{{.+}} : f16 to f32
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.fp8.f32
// CHECK-COUNT-4: arith.extf %{{.+}} : f16 to f32
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.bf8.f32

// -----

module {
  func.func @intr_bf16_to_4f8(%arg0: bf16, %arg1: bf16, %arg2: bf16, %arg3: bf16) -> (vector<4xf8E4M3FNUZ>, vector<4xf8E5M2FNUZ>) {
    %a0 = arith.truncf %arg0 : bf16 to f8E4M3FNUZ
    %a1 = arith.truncf %arg1 : bf16 to f8E4M3FNUZ
    %a2 = arith.truncf %arg2 : bf16 to f8E4M3FNUZ
    %a3 = arith.truncf %arg3 : bf16 to f8E4M3FNUZ
    %b0 = arith.truncf %arg0 : bf16 to f8E5M2FNUZ
    %b1 = arith.truncf %arg1 : bf16 to f8E5M2FNUZ
    %b2 = arith.truncf %arg2 : bf16 to f8E5M2FNUZ
    %b3 = arith.truncf %arg3 : bf16 to f8E5M2FNUZ
    %a_init = arith.constant dense<0.000000e+00> : vector<4xf8E4M3FNUZ>
    %b_init = arith.constant dense<0.000000e+00> : vector<4xf8E5M2FNUZ>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = vector.insert %a0, %a_init [%c0] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %1 = vector.insert %a1, %0 [%c1] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %2 = vector.insert %a2, %1 [%c2] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %a = vector.insert %a3, %2 [%c3] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %3 = vector.insert %b0, %b_init [%c0] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %4 = vector.insert %b1, %3 [%c1] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %5 = vector.insert %b2, %4 [%c2] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %b = vector.insert %b3, %5 [%c3] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    return %a, %b : vector<4xf8E4M3FNUZ>, vector<4xf8E5M2FNUZ>
  }
}

// CHECK-LABEL: @intr_bf16_to_4f8
// CHECK-COUNT-4: arith.extf %{{.+}} : bf16 to f32
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.fp8.f32
// CHECK-COUNT-4: arith.extf %{{.+}} : bf16 to f32
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.bf8.f32

// -----

module {
  func.func @intr_f32_to_4f8(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32) -> (vector<4xf8E4M3FNUZ>, vector<4xf8E5M2FNUZ>) {
    %a0 = arith.truncf %arg0 : f32 to f8E4M3FNUZ
    %a1 = arith.truncf %arg1 : f32 to f8E4M3FNUZ
    %a2 = arith.truncf %arg2 : f32 to f8E4M3FNUZ
    %a3 = arith.truncf %arg3 : f32 to f8E4M3FNUZ
    %b0 = arith.truncf %arg0 : f32 to f8E5M2FNUZ
    %b1 = arith.truncf %arg1 : f32 to f8E5M2FNUZ
    %b2 = arith.truncf %arg2 : f32 to f8E5M2FNUZ
    %b3 = arith.truncf %arg3 : f32 to f8E5M2FNUZ
    %a_init = arith.constant dense<0.000000e+00> : vector<4xf8E4M3FNUZ>
    %b_init = arith.constant dense<0.000000e+00> : vector<4xf8E5M2FNUZ>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = vector.insert %a0, %a_init [%c0] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %1 = vector.insert %a1, %0 [%c1] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %2 = vector.insert %a2, %1 [%c2] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %a = vector.insert %a3, %2 [%c3] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %3 = vector.insert %b0, %b_init [%c0] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %4 = vector.insert %b1, %3 [%c1] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %5 = vector.insert %b2, %4 [%c2] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %b = vector.insert %b3, %5 [%c3] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    return %a, %b : vector<4xf8E4M3FNUZ>, vector<4xf8E5M2FNUZ>
  }
}

// CHECK-LABEL: @intr_f32_to_4f8
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.fp8.f32
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.bf8.f32

// -----

module {
  func.func @intr_f64_to_4f8(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: f64) -> (vector<4xf8E4M3FNUZ>, vector<4xf8E5M2FNUZ>) {
    %a0 = arith.truncf %arg0 : f64 to f8E4M3FNUZ
    %a1 = arith.truncf %arg1 : f64 to f8E4M3FNUZ
    %a2 = arith.truncf %arg2 : f64 to f8E4M3FNUZ
    %a3 = arith.truncf %arg3 : f64 to f8E4M3FNUZ
    %b0 = arith.truncf %arg0 : f64 to f8E5M2FNUZ
    %b1 = arith.truncf %arg1 : f64 to f8E5M2FNUZ
    %b2 = arith.truncf %arg2 : f64 to f8E5M2FNUZ
    %b3 = arith.truncf %arg3 : f64 to f8E5M2FNUZ
    %a_init = arith.constant dense<0.000000e+00> : vector<4xf8E4M3FNUZ>
    %b_init = arith.constant dense<0.000000e+00> : vector<4xf8E5M2FNUZ>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = vector.insert %a0, %a_init [%c0] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %1 = vector.insert %a1, %0 [%c1] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %2 = vector.insert %a2, %1 [%c2] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %a = vector.insert %a3, %2 [%c3] : f8E4M3FNUZ into vector<4xf8E4M3FNUZ>
    %3 = vector.insert %b0, %b_init [%c0] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %4 = vector.insert %b1, %3 [%c1] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %5 = vector.insert %b2, %4 [%c2] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    %b = vector.insert %b3, %5 [%c3] : f8E5M2FNUZ into vector<4xf8E5M2FNUZ>
    return %a, %b : vector<4xf8E4M3FNUZ>, vector<4xf8E5M2FNUZ>
  }
}

// CHECK-LABEL: @intr_f64_to_4f8
// CHECK-COUNT-4: arith.truncf %{{.+}} : f64 to f32
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.fp8.f32
// CHECK-COUNT-4: arith.truncf %{{.+}} : f64 to f32
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.bf8.f32

// -----

module {
  func.func @intr_4f8_to_f16(%arg0: vector<4xf8E4M3FNUZ>, %arg1: vector<4xf8E5M2FNUZ>) -> (f16, f16, f16, f16, f16, f16, f16, f16) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %a0 = vector.extract %arg0[%c0] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a1 = vector.extract %arg0[%c1] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a2 = vector.extract %arg0[%c2] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a3 = vector.extract %arg0[%c3] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %b0 = vector.extract %arg1[%c0] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b1 = vector.extract %arg1[%c1] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b2 = vector.extract %arg1[%c2] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b3 = vector.extract %arg1[%c3] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %0 = arith.extf %a0 : f8E4M3FNUZ to f16
    %1 = arith.extf %a1 : f8E4M3FNUZ to f16
    %2 = arith.extf %a2 : f8E4M3FNUZ to f16
    %3 = arith.extf %a3 : f8E4M3FNUZ to f16
    %4 = arith.extf %b0 : f8E5M2FNUZ to f16
    %5 = arith.extf %b1 : f8E5M2FNUZ to f16
    %6 = arith.extf %b2 : f8E5M2FNUZ to f16
    %7 = arith.extf %b3 : f8E5M2FNUZ to f16
    return %0, %1, %2, %3, %4, %5, %6, %7 : f16, f16, f16, f16, f16, f16, f16, f16
  }
}

// CHECK-LABEL: @intr_4f8_to_f16
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.f32.fp8
// CHECK-COUNT-2: llvm.amdgcn.cvt.pkrtz
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.f32.bf8
// CHECK-COUNT-2: llvm.amdgcn.cvt.pkrtz

// -----

module {
  func.func @intr_4f8_to_bf16(%arg0: vector<4xf8E4M3FNUZ>, %arg1: vector<4xf8E5M2FNUZ>) -> (bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %a0 = vector.extract %arg0[%c0] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a1 = vector.extract %arg0[%c1] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a2 = vector.extract %arg0[%c2] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a3 = vector.extract %arg0[%c3] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %b0 = vector.extract %arg1[%c0] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b1 = vector.extract %arg1[%c1] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b2 = vector.extract %arg1[%c2] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b3 = vector.extract %arg1[%c3] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %0 = arith.extf %a0 : f8E4M3FNUZ to bf16
    %1 = arith.extf %a1 : f8E4M3FNUZ to bf16
    %2 = arith.extf %a2 : f8E4M3FNUZ to bf16
    %3 = arith.extf %a3 : f8E4M3FNUZ to bf16
    %4 = arith.extf %b0 : f8E5M2FNUZ to bf16
    %5 = arith.extf %b1 : f8E5M2FNUZ to bf16
    %6 = arith.extf %b2 : f8E5M2FNUZ to bf16
    %7 = arith.extf %b3 : f8E5M2FNUZ to bf16
    return %0, %1, %2, %3, %4, %5, %6, %7 : bf16, bf16, bf16, bf16, bf16, bf16, bf16, bf16
  }
}

// CHECK-LABEL: @intr_4f8_to_bf16
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.f32.fp8
// CHECK-COUNT-8: llvm.bitcast
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.f32.bf8
// CHECK-COUNT-8: llvm.bitcast
// CHECK-NOT: arith.truncf %{{.+}} : f32 to bf16

// -----

module {
  func.func @intr_4f8_to_f32(%arg0: vector<4xf8E4M3FNUZ>, %arg1: vector<4xf8E5M2FNUZ>) -> (f32, f32, f32, f32, f32, f32, f32, f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %a0 = vector.extract %arg0[%c0] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a1 = vector.extract %arg0[%c1] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a2 = vector.extract %arg0[%c2] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a3 = vector.extract %arg0[%c3] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %b0 = vector.extract %arg1[%c0] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b1 = vector.extract %arg1[%c1] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b2 = vector.extract %arg1[%c2] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b3 = vector.extract %arg1[%c3] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %0 = arith.extf %a0 : f8E4M3FNUZ to f32
    %1 = arith.extf %a1 : f8E4M3FNUZ to f32
    %2 = arith.extf %a2 : f8E4M3FNUZ to f32
    %3 = arith.extf %a3 : f8E4M3FNUZ to f32
    %4 = arith.extf %b0 : f8E5M2FNUZ to f32
    %5 = arith.extf %b1 : f8E5M2FNUZ to f32
    %6 = arith.extf %b2 : f8E5M2FNUZ to f32
    %7 = arith.extf %b3 : f8E5M2FNUZ to f32
    return %0, %1, %2, %3, %4, %5, %6, %7 : f32, f32, f32, f32, f32, f32, f32, f32
  }
}

// CHECK-LABEL: @intr_4f8_to_f32
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.f32.fp8
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.f32.bf8

// -----

module {
  func.func @intr_4f8_to_f64(%arg0: vector<4xf8E4M3FNUZ>, %arg1: vector<4xf8E5M2FNUZ>) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %a0 = vector.extract %arg0[%c0] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a1 = vector.extract %arg0[%c1] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a2 = vector.extract %arg0[%c2] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %a3 = vector.extract %arg0[%c3] : f8E4M3FNUZ from vector<4xf8E4M3FNUZ>
    %b0 = vector.extract %arg1[%c0] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b1 = vector.extract %arg1[%c1] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b2 = vector.extract %arg1[%c2] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %b3 = vector.extract %arg1[%c3] : f8E5M2FNUZ from vector<4xf8E5M2FNUZ>
    %0 = arith.extf %a0 : f8E4M3FNUZ to f64
    %1 = arith.extf %a1 : f8E4M3FNUZ to f64
    %2 = arith.extf %a2 : f8E4M3FNUZ to f64
    %3 = arith.extf %a3 : f8E4M3FNUZ to f64
    %4 = arith.extf %b0 : f8E5M2FNUZ to f64
    %5 = arith.extf %b1 : f8E5M2FNUZ to f64
    %6 = arith.extf %b2 : f8E5M2FNUZ to f64
    %7 = arith.extf %b3 : f8E5M2FNUZ to f64
    return %0, %1, %2, %3, %4, %5, %6, %7 : f64, f64, f64, f64, f64, f64, f64, f64
  }
}

// CHECK-LABEL: @intr_4f8_to_f64
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.f32.fp8
// CHECK-COUNT-4: arith.extf %{{.+}} : f32 to f64
// CHECK-COUNT-2: llvm.amdgcn.cvt.pk.f32.bf8
// CHECK-COUNT-4: arith.extf %{{.+}} : f32 to f64

// -----

module {
  func.func @intr_f32_to_f4(%arg0: f32) -> f4E2M1FN {
    %a = arith.truncf %arg0 : f32 to f4E2M1FN
    return %a : f4E2M1FN
  }
}

// CHECK-LABEL: @intr_f32_to_f4
// CHECK-NOT: llvm.amdgcn.cvt.scalef32.pk.fp4.f32
// CHECK: arith.truncf %{{.+}} : f32 to f4E2M1FN

// GFX950-LABEL: @intr_f32_to_f4
// GFX950-DAG: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
// GFX950-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// GFX950-DAG: %[[NEG_ZERO:.*]] = llvm.mlir.constant(-0.000000e+00 : f32) : f32
// GFX950: %[[IS_NAN:.*]] = arith.cmpf uno, %arg0, %arg0 : f32
// GFX950: %[[X:.*]] = arith.select %[[IS_NAN]], %[[NEG_ZERO]], %arg0 : f32
// GFX950: %[[CVT:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.scalef32.pk.fp4.f32"(%[[ZERO]], %[[X]], %[[X]], %[[ONE]], %[[ZERO]])
// GFX950: %[[BITS:.*]] = llvm.trunc %[[CVT]] : i32 to i4
// GFX950: arith.bitcast %[[BITS]] : i4 to f4E2M1FN

// -----

module {
  func.func @intr_bf16_to_f4(%arg0: bf16) -> f4E2M1FN {
    %a = arith.truncf %arg0 : bf16 to f4E2M1FN
    return %a : f4E2M1FN
  }
}

// CHECK-LABEL: @intr_bf16_to_f4
// CHECK-NOT: llvm.amdgcn.cvt.scalef32.pk.fp4.f32
// CHECK: arith.truncf %{{.+}} : bf16 to f4E2M1FN

// GFX950-LABEL: @intr_bf16_to_f4
// GFX950: %[[X:.*]] = arith.extf %arg0 : bf16 to f32
// GFX950: arith.cmpf uno, %[[X]], %[[X]] : f32
// GFX950: llvm.call_intrinsic "llvm.amdgcn.cvt.scalef32.pk.fp4.f32"

// -----

module {
  func.func @intr_f16_to_f4(%arg0: f16) -> f4E2M1FN {
    %a = arith.truncf %arg0 : f16 to f4E2M1FN
    return %a : f4E2M1FN
  }
}

// CHECK-LABEL: @intr_f16_to_f4
// CHECK-NOT: llvm.amdgcn.cvt.scalef32.pk.fp4.f32
// CHECK: arith.truncf %{{.+}} : f16 to f4E2M1FN

// GFX950-LABEL: @intr_f16_to_f4
// GFX950: %[[X:.*]] = arith.extf %arg0 : f16 to f32
// GFX950: arith.cmpf uno, %[[X]], %[[X]] : f32
// GFX950: llvm.call_intrinsic "llvm.amdgcn.cvt.scalef32.pk.fp4.f32"

// -----

module {
  func.func @intr_f64_to_f4(%arg0: f64) -> f4E2M1FN {
    %a = arith.truncf %arg0 : f64 to f4E2M1FN
    return %a : f4E2M1FN
  }
}

// CHECK-LABEL: @intr_f64_to_f4
// CHECK-NOT: llvm.amdgcn.cvt.scalef32.pk.fp4.f32
// CHECK: arith.truncf %{{.+}} : f64 to f4E2M1FN

// GFX950-LABEL: @intr_f64_to_f4
// GFX950-NOT: llvm.amdgcn.cvt.scalef32.pk.fp4.f32
// GFX950: arith.truncf %{{.+}} : f64 to f4E2M1FN

// -----

module {
  func.func @intr_f32_to_2f4(%arg0: f32, %arg1: f32) -> vector<2xf4E2M1FN> {
    %a0 = arith.truncf %arg0 : f32 to f4E2M1FN
    %a1 = arith.truncf %arg1 : f32 to f4E2M1FN
    %v = vector.from_elements %a0, %a1 : vector<2xf4E2M1FN>
    return %v : vector<2xf4E2M1FN>
  }
}

// CHECK-LABEL: @intr_f32_to_2f4
// CHECK-NOT: llvm.amdgcn.cvt.scalef32.pk.fp4.f32
// CHECK-COUNT-2: arith.truncf %{{.+}} : f32 to f4E2M1FN

// GFX950-LABEL: @intr_f32_to_2f4
// GFX950-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// GFX950-DAG: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
// GFX950: %[[X0:.*]] = arith.select %{{.+}}, %{{.+}}, %arg0 : f32
// GFX950: %[[X1:.*]] = arith.select %{{.+}}, %{{.+}}, %arg1 : f32
// GFX950: %[[CVT:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.scalef32.pk.fp4.f32"(%[[ZERO]], %[[X0]], %[[X1]], %[[ONE]], %[[ZERO]])
// GFX950: %[[BYTE:.*]] = llvm.trunc %[[CVT]] : i32 to i8
// GFX950: %[[BITS:.*]] = llvm.bitcast %[[BYTE]] : i8 to vector<2xi4>
// GFX950: builtin.unrealized_conversion_cast %[[BITS]] : vector<2xi4> to vector<2xf4E2M1FN>
// GFX950-NOT: llvm.amdgcn.cvt.scalef32.pk.fp4.f32

// -----

module {
  func.func @intr_bf16_to_8f4(%arg0: bf16, %arg1: bf16, %arg2: bf16, %arg3: bf16, %arg4: bf16, %arg5: bf16, %arg6: bf16, %arg7: bf16) -> vector<8xf4E2M1FN> {
    %a0 = arith.truncf %arg0 : bf16 to f4E2M1FN
    %a1 = arith.truncf %arg1 : bf16 to f4E2M1FN
    %a2 = arith.truncf %arg2 : bf16 to f4E2M1FN
    %a3 = arith.truncf %arg3 : bf16 to f4E2M1FN
    %a4 = arith.truncf %arg4 : bf16 to f4E2M1FN
    %a5 = arith.truncf %arg5 : bf16 to f4E2M1FN
    %a6 = arith.truncf %arg6 : bf16 to f4E2M1FN
    %a7 = arith.truncf %arg7 : bf16 to f4E2M1FN
    %v = vector.from_elements %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7 : vector<8xf4E2M1FN>
    return %v : vector<8xf4E2M1FN>
  }
}

// GFX950-LABEL: @intr_bf16_to_8f4
// GFX950-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// GFX950-DAG: %[[BYTE1:.*]] = llvm.mlir.constant(1 : i32) : i32
// GFX950-DAG: %[[BYTE2:.*]] = llvm.mlir.constant(2 : i32) : i32
// GFX950-DAG: %[[BYTE3:.*]] = llvm.mlir.constant(3 : i32) : i32
// GFX950: %[[W0:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.scalef32.pk.fp4.f32"(%[[ZERO]], %{{.+}}, %{{.+}}, %{{.+}}, %[[ZERO]])
// GFX950: %[[W1:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.scalef32.pk.fp4.f32"(%[[W0]], %{{.+}}, %{{.+}}, %{{.+}}, %[[BYTE1]])
// GFX950: %[[W2:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.scalef32.pk.fp4.f32"(%[[W1]], %{{.+}}, %{{.+}}, %{{.+}}, %[[BYTE2]])
// GFX950: %[[W3:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.scalef32.pk.fp4.f32"(%[[W2]], %{{.+}}, %{{.+}}, %{{.+}}, %[[BYTE3]])
// GFX950: %[[BITS:.*]] = llvm.bitcast %[[W3]] : i32 to vector<8xi4>
// GFX950: builtin.unrealized_conversion_cast %[[BITS]] : vector<8xi4> to vector<8xf4E2M1FN>

// -----

module {
  func.func @intr_f16_to_16f4(%arg0: f16, %arg1: f16, %arg2: f16, %arg3: f16, %arg4: f16, %arg5: f16, %arg6: f16, %arg7: f16, %arg8: f16, %arg9: f16, %arg10: f16, %arg11: f16, %arg12: f16, %arg13: f16, %arg14: f16, %arg15: f16) -> vector<16xf4E2M1FN> {
    %a0 = arith.truncf %arg0 : f16 to f4E2M1FN
    %a1 = arith.truncf %arg1 : f16 to f4E2M1FN
    %a2 = arith.truncf %arg2 : f16 to f4E2M1FN
    %a3 = arith.truncf %arg3 : f16 to f4E2M1FN
    %a4 = arith.truncf %arg4 : f16 to f4E2M1FN
    %a5 = arith.truncf %arg5 : f16 to f4E2M1FN
    %a6 = arith.truncf %arg6 : f16 to f4E2M1FN
    %a7 = arith.truncf %arg7 : f16 to f4E2M1FN
    %a8 = arith.truncf %arg8 : f16 to f4E2M1FN
    %a9 = arith.truncf %arg9 : f16 to f4E2M1FN
    %a10 = arith.truncf %arg10 : f16 to f4E2M1FN
    %a11 = arith.truncf %arg11 : f16 to f4E2M1FN
    %a12 = arith.truncf %arg12 : f16 to f4E2M1FN
    %a13 = arith.truncf %arg13 : f16 to f4E2M1FN
    %a14 = arith.truncf %arg14 : f16 to f4E2M1FN
    %a15 = arith.truncf %arg15 : f16 to f4E2M1FN
    %v = vector.from_elements %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7, %a8, %a9, %a10, %a11, %a12, %a13, %a14, %a15 : vector<16xf4E2M1FN>
    return %v : vector<16xf4E2M1FN>
  }
}

// GFX950-LABEL: @intr_f16_to_16f4
// GFX950-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// GFX950-DAG: %[[BYTE1:.*]] = llvm.mlir.constant(1 : i32) : i32
// GFX950-DAG: %[[BYTE2:.*]] = llvm.mlir.constant(2 : i32) : i32
// GFX950-DAG: %[[BYTE3:.*]] = llvm.mlir.constant(3 : i32) : i32
// GFX950-DAG: %[[UNDEF:.*]] = llvm.mlir.undef : vector<2xi32>
// GFX950: %[[W0:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.scalef32.pk.fp4.f32"(%[[ZERO]], %{{.+}}, %{{.+}}, %{{.+}}, %[[ZERO]])
// GFX950: %[[W1:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.scalef32.pk.fp4.f32"(%[[W0]], %{{.+}}, %{{.+}}, %{{.+}}, %[[BYTE1]])
// GFX950: %[[W2:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.scalef32.pk.fp4.f32"(%[[W1]], %{{.+}}, %{{.+}}, %{{.+}}, %[[BYTE2]])
// GFX950: %[[W3:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.scalef32.pk.fp4.f32"(%[[W2]], %{{.+}}, %{{.+}}, %{{.+}}, %[[BYTE3]])
// GFX950: %[[V0:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.scalef32.pk.fp4.f32"(%[[ZERO]], %{{.+}}, %{{.+}}, %{{.+}}, %[[ZERO]])
// GFX950: %[[V1:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.scalef32.pk.fp4.f32"(%[[V0]], %{{.+}}, %{{.+}}, %{{.+}}, %[[BYTE1]])
// GFX950: %[[V2:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.scalef32.pk.fp4.f32"(%[[V1]], %{{.+}}, %{{.+}}, %{{.+}}, %[[BYTE2]])
// GFX950: %[[V3:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.scalef32.pk.fp4.f32"(%[[V2]], %{{.+}}, %{{.+}}, %{{.+}}, %[[BYTE3]])
// GFX950: %[[LO:.*]] = llvm.insertelement %[[W3]], %[[UNDEF]][%[[ZERO]] : i32] : vector<2xi32>
// GFX950: %[[WORDS:.*]] = llvm.insertelement %[[V3]], %[[LO]][%[[BYTE1]] : i32] : vector<2xi32>
// GFX950: %[[BITS:.*]] = llvm.bitcast %[[WORDS]] : vector<2xi32> to vector<16xi4>
// GFX950: builtin.unrealized_conversion_cast %[[BITS]] : vector<16xi4> to vector<16xf4E2M1FN>

// -----

module {
  func.func @intr_f8_to_f4(%arg0: f8E4M3FN) -> f4E2M1FN {
    %a = arith.truncf %arg0 : f8E4M3FN to f4E2M1FN
    return %a : f4E2M1FN
  }
}

// CHECK-LABEL: @intr_f8_to_f4
// CHECK: arith.truncf %{{.+}} : f8E4M3FN to f4E2M1FN

// GFX950-LABEL: @intr_f8_to_f4
// GFX950-NOT: llvm.amdgcn
// GFX950: arith.truncf %{{.+}} : f8E4M3FN to f4E2M1FN

// -----

module {
  func.func @intr_bf16_to_4f8_from_elements(%arg0: bf16, %arg1: bf16, %arg2: bf16, %arg3: bf16) -> vector<4xf8E4M3FNUZ> {
    %a0 = arith.truncf %arg0 : bf16 to f8E4M3FNUZ
    %a1 = arith.truncf %arg1 : bf16 to f8E4M3FNUZ
    %a2 = arith.truncf %arg2 : bf16 to f8E4M3FNUZ
    %a3 = arith.truncf %arg3 : bf16 to f8E4M3FNUZ
    %v = vector.from_elements %a0, %a1, %a2, %a3 : vector<4xf8E4M3FNUZ>
    return %v : vector<4xf8E4M3FNUZ>
  }
}

// CHECK-LABEL: @intr_bf16_to_4f8_from_elements
// CHECK-DAG: %[[TRUE:.*]] = llvm.mlir.constant(true) : i1
// CHECK-DAG: %[[FALSE:.*]] = llvm.mlir.constant(false) : i1
// CHECK-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[X0:.*]] = arith.extf %arg0 : bf16 to f32
// CHECK: %[[X1:.*]] = arith.extf %arg1 : bf16 to f32
// CHECK: %[[X2:.*]] = arith.extf %arg2 : bf16 to f32
// CHECK: %[[X3:.*]] = arith.extf %arg3 : bf16 to f32
// CHECK: %[[LO:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.pk.fp8.f32"(%[[X0]], %[[X1]], %{{.+}}, %[[FALSE]])
// CHECK: %[[W0:.*]] = llvm.insertelement %[[LO]], %{{.+}}[%[[ZERO]] : i32] : vector<1xi32>
// CHECK: %[[OLD:.*]] = llvm.extractelement %[[W0]][%[[ZERO]] : i32] : vector<1xi32>
// CHECK: %[[HI:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.pk.fp8.f32"(%[[X2]], %[[X3]], %[[OLD]], %[[TRUE]])
// CHECK: %[[W1:.*]] = llvm.insertelement %[[HI]], %[[W0]][%[[ZERO]] : i32] : vector<1xi32>
// CHECK: %[[BYTES:.*]] = llvm.bitcast %[[W1]] : vector<1xi32> to vector<4xi8>
// CHECK: builtin.unrealized_conversion_cast %[[BYTES]] : vector<4xi8> to vector<4xf8E4M3FNUZ>
// CHECK-NOT: vector.from_elements

// -----

module {
  func.func @intr_f16_to_8bf8_from_elements(%arg0: f16, %arg1: f16, %arg2: f16, %arg3: f16, %arg4: f16, %arg5: f16, %arg6: f16, %arg7: f16) -> vector<8xf8E5M2FNUZ> {
    %a0 = arith.truncf %arg0 : f16 to f8E5M2FNUZ
    %a1 = arith.truncf %arg1 : f16 to f8E5M2FNUZ
    %a2 = arith.truncf %arg2 : f16 to f8E5M2FNUZ
    %a3 = arith.truncf %arg3 : f16 to f8E5M2FNUZ
    %a4 = arith.truncf %arg4 : f16 to f8E5M2FNUZ
    %a5 = arith.truncf %arg5 : f16 to f8E5M2FNUZ
    %a6 = arith.truncf %arg6 : f16 to f8E5M2FNUZ
    %a7 = arith.truncf %arg7 : f16 to f8E5M2FNUZ
    %v = vector.from_elements %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7 : vector<8xf8E5M2FNUZ>
    return %v : vector<8xf8E5M2FNUZ>
  }
}

// CHECK-LABEL: @intr_f16_to_8bf8_from_elements
// CHECK-COUNT-4: llvm.call_intrinsic "llvm.amdgcn.cvt.pk.bf8.f32"
// CHECK: %[[WORDS:.*]] = llvm.insertelement
// CHECK: llvm.bitcast %[[WORDS]] : vector<2xi32> to vector<8xi8>

// -----

module {
  func.func @intr_f32_to_2f8_from_elements(%arg0: f32, %arg1: f32) -> vector<2xf8E4M3FN> {
    %a0 = arith.truncf %arg0 : f32 to f8E4M3FN
    %a1 = arith.truncf %arg1 : f32 to f8E4M3FN
    %v = vector.from_elements %a0, %a1 : vector<2xf8E4M3FN>
    return %v : vector<2xf8E4M3FN>
  }
}

// GFX950-LABEL: @intr_f32_to_2f8_from_elements
// GFX950-DAG: %[[FALSE:.*]] = llvm.mlir.constant(false) : i1
// GFX950: %[[PAIR:.*]] = llvm.call_intrinsic "llvm.amdgcn.cvt.pk.fp8.f32"(%arg0, %arg1, %{{.+}}, %[[FALSE]])
// GFX950: %[[WORD:.*]] = llvm.insertelement %[[PAIR]]
// GFX950: %[[HALVES:.*]] = llvm.bitcast %[[WORD]] : vector<1xi32> to vector<2xi16>
// GFX950: %[[HALF:.*]] = llvm.extractelement %[[HALVES]]
// GFX950: %[[BYTES:.*]] = llvm.bitcast %[[HALF]] : i16 to vector<2xi8>
// GFX950: builtin.unrealized_conversion_cast %[[BYTES]] : vector<2xi8> to vector<2xf8E4M3FN>
// GFX950-NOT: vector.from_elements

// -----

module {
  func.func @intr_bf16_to_8f8_from_elements(%arg0: bf16, %arg1: bf16, %arg2: bf16, %arg3: bf16, %arg4: bf16, %arg5: bf16, %arg6: bf16, %arg7: bf16) -> vector<8xf8E4M3FN> {
    %a0 = arith.truncf %arg0 : bf16 to f8E4M3FN
    %a1 = arith.truncf %arg1 : bf16 to f8E4M3FN
    %a2 = arith.truncf %arg2 : bf16 to f8E4M3FN
    %a3 = arith.truncf %arg3 : bf16 to f8E4M3FN
    %a4 = arith.truncf %arg4 : bf16 to f8E4M3FN
    %a5 = arith.truncf %arg5 : bf16 to f8E4M3FN
    %a6 = arith.truncf %arg6 : bf16 to f8E4M3FN
    %a7 = arith.truncf %arg7 : bf16 to f8E4M3FN
    %v = vector.from_elements %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7 : vector<8xf8E4M3FN>
    return %v : vector<8xf8E4M3FN>
  }
}

// GFX950-LABEL: @intr_bf16_to_8f8_from_elements
// GFX950-DAG: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// GFX950-DAG: %[[TRUE:.*]] = llvm.mlir.constant(true) : i1
// GFX950-DAG: %[[FALSE:.*]] = llvm.mlir.constant(false) : i1
// GFX950-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// GFX950: %[[X0:.*]] = arith.extf %arg0 : bf16 to f32
// GFX950: %[[X1:.*]] = arith.extf %arg1 : bf16 to f32
// GFX950: %[[X2:.*]] = arith.extf %arg2 : bf16 to f32
// GFX950: %[[X3:.*]] = arith.extf %arg3 : bf16 to f32
// GFX950: %[[X4:.*]] = arith.extf %arg4 : bf16 to f32
// GFX950: %[[X5:.*]] = arith.extf %arg5 : bf16 to f32
// GFX950: %[[X6:.*]] = arith.extf %arg6 : bf16 to f32
// GFX950: %[[X7:.*]] = arith.extf %arg7 : bf16 to f32
// GFX950: llvm.call_intrinsic "llvm.amdgcn.cvt.pk.fp8.f32"(%[[X0]], %[[X1]], %{{.+}}, %[[FALSE]])
// GFX950: llvm.insertelement %{{.+}}, %{{.+}}[%[[ZERO]] : i32] : vector<2xi32>
// GFX950: llvm.call_intrinsic "llvm.amdgcn.cvt.pk.fp8.f32"(%[[X2]], %[[X3]], %{{.+}}, %[[TRUE]])
// GFX950: llvm.insertelement %{{.+}}, %{{.+}}[%[[ZERO]] : i32] : vector<2xi32>
// GFX950: llvm.call_intrinsic "llvm.amdgcn.cvt.pk.fp8.f32"(%[[X4]], %[[X5]], %{{.+}}, %[[FALSE]])
// GFX950: llvm.insertelement %{{.+}}, %{{.+}}[%[[ONE]] : i32] : vector<2xi32>
// GFX950: llvm.call_intrinsic "llvm.amdgcn.cvt.pk.fp8.f32"(%[[X6]], %[[X7]], %{{.+}}, %[[TRUE]])
// GFX950: %[[WORDS:.*]] = llvm.insertelement %{{.+}}, %{{.+}}[%[[ONE]] : i32] : vector<2xi32>
// GFX950: %[[BYTES:.*]] = llvm.bitcast %[[WORDS]] : vector<2xi32> to vector<8xi8>
// GFX950: builtin.unrealized_conversion_cast %[[BYTES]] : vector<8xi8> to vector<8xf8E4M3FN>
// GFX950-NOT: vector.from_elements

// -----

module {
  func.func @intr_f16_to_16bf8_from_elements(%arg0: f16, %arg1: f16, %arg2: f16, %arg3: f16, %arg4: f16, %arg5: f16, %arg6: f16, %arg7: f16, %arg8: f16, %arg9: f16, %arg10: f16, %arg11: f16, %arg12: f16, %arg13: f16, %arg14: f16, %arg15: f16) -> vector<16xf8E5M2> {
    %a0 = arith.truncf %arg0 : f16 to f8E5M2
    %a1 = arith.truncf %arg1 : f16 to f8E5M2
    %a2 = arith.truncf %arg2 : f16 to f8E5M2
    %a3 = arith.truncf %arg3 : f16 to f8E5M2
    %a4 = arith.truncf %arg4 : f16 to f8E5M2
    %a5 = arith.truncf %arg5 : f16 to f8E5M2
    %a6 = arith.truncf %arg6 : f16 to f8E5M2
    %a7 = arith.truncf %arg7 : f16 to f8E5M2
    %a8 = arith.truncf %arg8 : f16 to f8E5M2
    %a9 = arith.truncf %arg9 : f16 to f8E5M2
    %a10 = arith.truncf %arg10 : f16 to f8E5M2
    %a11 = arith.truncf %arg11 : f16 to f8E5M2
    %a12 = arith.truncf %arg12 : f16 to f8E5M2
    %a13 = arith.truncf %arg13 : f16 to f8E5M2
    %a14 = arith.truncf %arg14 : f16 to f8E5M2
    %a15 = arith.truncf %arg15 : f16 to f8E5M2
    %v = vector.from_elements %a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7, %a8, %a9, %a10, %a11, %a12, %a13, %a14, %a15 : vector<16xf8E5M2>
    return %v : vector<16xf8E5M2>
  }
}

// GFX950-LABEL: @intr_f16_to_16bf8_from_elements
// GFX950-COUNT-8: llvm.call_intrinsic "llvm.amdgcn.cvt.pk.bf8.f32"
// GFX950: %[[WORDS:.*]] = llvm.insertelement
// GFX950: %[[BYTES:.*]] = llvm.bitcast %[[WORDS]] : vector<4xi32> to vector<16xi8>
// GFX950: builtin.unrealized_conversion_cast %[[BYTES]] : vector<16xi8> to vector<16xf8E5M2>
// GFX950-NOT: vector.from_elements

// -----

module {
  func.func @intr_mixed_from_elements(%arg0: bf16, %arg1: f8E4M3FN) -> vector<2xf8E4M3FN> {
    %a0 = arith.truncf %arg0 : bf16 to f8E4M3FN
    %v = vector.from_elements %a0, %arg1 : vector<2xf8E4M3FN>
    return %v : vector<2xf8E4M3FN>
  }
}

// GFX950-LABEL: @intr_mixed_from_elements
// GFX950: llvm.call_intrinsic "llvm.amdgcn.cvt.pk.fp8.f32"
// GFX950: llvm.trunc %{{.+}} : i32 to i8
// GFX950: vector.from_elements %{{.+}}, %arg1 : vector<2xf8E4M3FN>

// -----

module {
  func.func @intr_bf16_to_3f8_from_elements(%arg0: bf16, %arg1: bf16, %arg2: bf16) -> vector<3xf8E4M3FN> {
    %a0 = arith.truncf %arg0 : bf16 to f8E4M3FN
    %a1 = arith.truncf %arg1 : bf16 to f8E4M3FN
    %a2 = arith.truncf %arg2 : bf16 to f8E4M3FN
    %v = vector.from_elements %a0, %a1, %a2 : vector<3xf8E4M3FN>
    return %v : vector<3xf8E4M3FN>
  }
}

// GFX950-LABEL: @intr_bf16_to_3f8_from_elements
// GFX950-COUNT-3: llvm.trunc %{{.+}} : i32 to i8
// GFX950: vector.from_elements %{{.+}}, %{{.+}}, %{{.+}} : vector<3xf8E4M3FN>
