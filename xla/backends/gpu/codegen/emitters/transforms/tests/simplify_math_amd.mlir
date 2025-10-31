// RUN: emitters_opt %s -split-input-file \
// RUN:   -xla-simplify-math \
// RUN:   -xla-lower-to-llvm="gpu_device_info='rocm_compute_capability {gcn_arch_name: \"gfx90a:sramecc+:xnack\"}'" \
// RUN: | FileCheck %s --check-prefix=LLVM-AMD

///////////////////////////////////////////////////////////////////////////////
// Test standalone pass: pow(2.0, x) -> exp2(x)

module {
  func.func @pow_of_two(%arg0: f32) -> (f32) {
    %c = arith.constant 2.0 : f32
    %0 = math.powf %c, %arg0 : f32
    return %0 : f32
  }
}

// LLVM-AMD-LABEL: @pow_of_two
// LLVM-AMD: llvm.call @__ocml_exp2_f32

// -----

///////////////////////////////////////////////////////////////////////////////
// Test standalone pass: pow(x, 1.0) -> x

module {
  func.func @pow_to_one(%arg0: f32) -> (f32) {
    %c = arith.constant 1.0 : f32
    %0 = math.powf %arg0, %c : f32
    return %0 : f32
  }
}

// LLVM-AMD-LABEL: @pow_to_one
// LLVM-AMD-NOT: llvm.call @__ocml_pow_f32
// LLVM-AMD-NOT: llvm.fmul
// LLVM-AMD: llvm.return %arg0

// -----

///////////////////////////////////////////////////////////////////////////////
// Test standalone pass: pow(x, 2.0) -> x * x

module {
  func.func @pow_to_two(%arg0: f32) -> (f32) {
    %c = arith.constant 2.0 : f32
    %0 = math.powf %arg0, %c : f32
    return %0 : f32
  }
}

// LLVM-AMD-LABEL: @pow_to_two
// LLVM-AMD-NOT: llvm.call @__ocml_pow_f32
// LLVM-AMD: [[RES:%.*]] = llvm.fmul %arg0, %arg0
// LLVM-AMD: llvm.return [[RES]]

// -----

///////////////////////////////////////////////////////////////////////////////
// Test standalone pass: pow(x, 3.0) -> x * x * x

module {
  func.func @pow_to_three(%arg0: f32) -> (f32) {
    %c = arith.constant 3.0 : f32
    %0 = math.powf %arg0, %c : f32
    return %0 : f32
  }
}

// LLVM-AMD-LABEL: @pow_to_three
// LLVM-AMD-NOT: llvm.call @__ocml_pow_f32
// LLVM-AMD: [[TMP:%.*]] = llvm.fmul %arg0, %arg0
// LLVM-AMD: [[RES:%.*]] = llvm.fmul %arg0, [[TMP]]
// LLVM-AMD: llvm.return [[RES]]

// -----

///////////////////////////////////////////////////////////////////////////////
// Test standalone pass: pow(x, -1.0) -> 1.0 / x

module {
  func.func @pow_to_minus_one(%arg0: f32) -> (f32) {
    %c = arith.constant -1.0 : f32
    %0 = math.powf %arg0, %c : f32
    return %0 : f32
  }
}

// LLVM-AMD-LABEL: @pow_to_minus_one
// LLVM-AMD-NOT: llvm.call @__ocml_pow_f32
// LLVM-AMD: [[ONE:%.*]] = llvm.mlir.constant(1.000000e+00 : f32)
// LLVM-AMD: [[RES:%.*]] = llvm.fdiv [[ONE]], %arg0
// LLVM-AMD: llvm.return [[RES]]

// -----

///////////////////////////////////////////////////////////////////////////////
// Test standalone pass: pow(x, 0.5) -> sqrt(x)

module {
  func.func @pow_to_half(%arg0: f32) -> (f32) {
    %c = arith.constant 5.000000e-01 : f32
    %0 = math.powf %arg0, %c : f32
    return %0 : f32
  }
}

// LLVM-AMD-LABEL: @pow_to_half
// LLVM-AMD-NOT: llvm.call @__ocml_pow_f32
// LLVM-AMD: [[RES:%.*]] = llvm.intr.sqrt(%arg0)
// LLVM-AMD: llvm.return [[RES]]

// -----

///////////////////////////////////////////////////////////////////////////////
// Test standalone pass: pow(x, -0.5) -> rsqrt(x)

module {
  func.func @pow_to_minus_half(%arg0: f32) -> (f32) {
    %c = arith.constant -5.000000e-01 : f32
    %0 = math.powf %arg0, %c : f32
    return %0 : f32
  }
}

// LLVM-AMD-LABEL: @pow_to_minus_half
// LLVM-AMD-NOT: llvm.call @__ocml_pow_f32
// LLVM-AMD: [[RES:%.*]] = llvm.call @__ocml_rsqrt_f32(%arg0)
// LLVM-AMD: llvm.return [[RES]]

// -----

///////////////////////////////////////////////////////////////////////////////
// Test standalone pass: pow(2.0^n, y) -> exp2(n * y) for n = -2

module {
  func.func @pow_of_quarter(%arg0: f32) -> (f32) {
    %c = arith.constant 2.500000e-01 : f32
    %0 = math.powf %c, %arg0 : f32
    return %0 : f32
  }
}

// LLVM-AMD-LABEL: @pow_of_quarter
// LLVM-AMD: [[MINUS_TWO:%.*]] = llvm.mlir.constant(-2.000000e+00 : f32)
// LLVM-AMD: [[SCALED:%.*]] = llvm.fmul %arg0, [[MINUS_TWO]]
// LLVM-AMD: [[RES:%.*]] = llvm.call @__ocml_exp2_f32([[SCALED]])
// LLVM-AMD: llvm.return [[RES]]

// -----

///////////////////////////////////////////////////////////////////////////////
// Test standalone pass: fpowi(x, 0) -> 1.0

module {
  func.func @fpowi_to_zero(%arg0: f32) -> (f32) {
    %c = arith.constant 0 : i32
    %0 = math.fpowi %arg0, %c : f32, i32
    return %0 : f32
  }
}

// LLVM-AMD-LABEL: @fpowi_to_zero
// LLVM-AMD-NOT: llvm.call @__ocml_pown_f32
// LLVM-AMD: [[ONE:%.*]] = llvm.mlir.constant(1.000000e+00 : f32)
// LLVM-AMD: llvm.return [[ONE]]

// -----

///////////////////////////////////////////////////////////////////////////////
// Test standalone pass: ipowi(x, 0) -> 1

module {
  func.func @ipowi_to_zero(%arg0: i32) -> (i32) {
    %c = arith.constant 0 : i32
    %0 = math.ipowi %arg0, %c : i32
    return %0 : i32
  }
}

// LLVM-AMD-LABEL: @ipowi_to_zero
// LLVM-AMD: [[ONE:%.*]] = llvm.mlir.constant(1 : i32)
// LLVM-AMD: llvm.return [[ONE]]

// -----

///////////////////////////////////////////////////////////////////////////////
// Test standalone pass: fpowi(x, positive_exponent) -> x * x * x * ...

module {
  func.func @fpowi_positive(%arg0: f32) -> (f32) {
    %c = arith.constant 3 : i32
    %0 = math.fpowi %arg0, %c : f32, i32
    return %0 : f32
  }
}

// LLVM-AMD-LABEL: @fpowi_positive
// LLVM-AMD-NOT: llvm.call @__ocml_pown_f32
// LLVM-AMD: [[TMP:%.*]] = llvm.fmul %arg0, %arg0
// LLVM-AMD: [[RES:%.*]] = llvm.fmul [[TMP]], %arg0
// LLVM-AMD: llvm.return [[RES]]

// -----

///////////////////////////////////////////////////////////////////////////////
// Test standalone pass: ipowi(x, positive_exponent) -> x * x * x * ...

module {
  func.func @ipowi_positive(%arg0: i32) -> (i32) {
    %c = arith.constant 3 : i32
    %0 = math.ipowi %arg0, %c : i32
    return %0 : i32
  }
}

// LLVM-AMD-LABEL: @ipowi_positive
// LLVM-AMD: [[TMP:%.*]] = llvm.mul %arg0, %arg0
// LLVM-AMD: [[RES:%.*]] = llvm.mul [[TMP]], %arg0
// LLVM-AMD: llvm.return [[RES]]
