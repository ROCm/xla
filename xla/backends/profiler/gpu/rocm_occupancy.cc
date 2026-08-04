/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/profiler/gpu/rocm_occupancy.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

#include "absl/log/log.h"

namespace xla {
namespace profiler {
namespace {

// Sentinel for "this resource does not bound the launch at all". Only ever
// compared with std::min against real workgroup counts, never divided by.
constexpr uint32_t kUnbounded = 0xFFFFFFFFu;

// Saturating, not wrapping. These inputs are not adversarial but they are not
// trustworthy either -- the version-skew guard below exists precisely because
// an SDK that predates a target hands back register counts whose encoding we
// do not understand. A wrapping AlignTo turns a huge count into 0, and 0 reads
// downstream as "no resource pressure": it divides by zero in the VGPR bound
// (a SIGFPE that would abort the *traced* process, the worst failure mode a
// profiler has) and reports unbounded workgroups in the LDS bound. Saturating
// turns garbage into "maximum pressure" instead, which the model already knows
// how to degrade -- to the 1-wave / 1-workgroup floor.
inline uint32_t AlignTo(uint32_t v, uint32_t a) {
  if (a == 0) return v;
  if (v > std::numeric_limits<uint32_t>::max() - (a - 1)) {
    return std::numeric_limits<uint32_t>::max();
  }
  return ((v + a - 1) / a) * a;
}

inline uint32_t SaturatingAdd(uint32_t a, uint32_t b) {
  return a > std::numeric_limits<uint32_t>::max() - b
             ? std::numeric_limits<uint32_t>::max()
             : a + b;
}

inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

inline uint32_t Clamp(uint32_t v, uint32_t lo, uint32_t hi) {
  return std::min(std::max(v, lo), hi);
}

// The FeatureGFX90AInsts family: gfx90a, gfx94x, gfx95x. LLVM keys almost
// every constant that matters here off that one subtarget feature, so the
// only differences between the three are the LDS pool size and its
// allocation granule.
constexpr AmdGpuTargetConstants MakeGfx90aFamily(const char* name,
                                                 uint32_t lds_per_cu,
                                                 uint32_t lds_granule) {
  return AmdGpuTargetConstants{
      /*name=*/name,
      /*max_waves_per_simd=*/8,  // getMaxWavesPerEU: isGFX90A -> 8
      /*simd_per_cu=*/4,         // getEUsPerCU: pre-GFX10 -> 4
      /*wave_front_size=*/64,
      /*total_vgprs=*/512,       // getTotalNumVGPRs: FeatureGFX90AInsts
      /*vgpr_granule=*/8,        // getVGPRAllocGranule: FeatureGFX90AInsts
      /*arch_vgpr_granule=*/4,   // getArchVGPRAllocGranule
      /*lds_per_cu=*/lds_per_cu,
      /*lds_granule=*/lds_granule,
      /*total_sgprs=*/800,       // AMDGPUTargetParser: Major >= 8 -> 800
      /*sgpr_granule=*/16,       // Major 8..9 -> 16
      /*sgpr_trap_reserve=*/0,   // no +trap-handler in normal HIP codegen
      /*max_barriers=*/16,       // getMaxWorkGroupsPerCU; never binds on CDNA
      /*sgpr_limited=*/true,     // isSGPROccupancyLimited: Major < 10
      /*unified_vgpr_file=*/true,
      /*exact=*/true};
}

}  // namespace

const char* OccupancyLimiterName(OccupancyLimiter l) {
  switch (l) {
    case OccupancyLimiter::kNone:
      return "none";
    case OccupancyLimiter::kVGPR:
      return "vgpr";
    case OccupancyLimiter::kLDS:
      return "lds";
    case OccupancyLimiter::kSGPR:
      return "sgpr";
    case OccupancyLimiter::kWorkgroup:
      return "workgroup";
    case OccupancyLimiter::kBarrier:
      return "barrier";
  }
  return "unknown";
}

std::optional<AmdGpuTargetConstants> LookupTargetConstants(uint32_t v) {
  const uint32_t major = (v / 10000) % 100;
  const uint32_t minor = (v / 100) % 100;
  const uint32_t step = v % 100;

  if (major != 9) {
    // gfx10/11/12 have wave32, a much larger VGPR file and different
    // granules. We have no validated constants for them; emit nothing rather
    // than a plausible-looking wrong number.
    return std::nullopt;
  }

  if (minor == 0 && step == 10) {  // gfx90a, MI200
    return MakeGfx90aFamily("gfx90a", /*lds_per_cu=*/65536,
                            /*lds_granule=*/512);
  }
  if (minor == 4) {  // gfx940/941/942; MI300A/MI300X/MI325
    return MakeGfx90aFamily("gfx94x", /*lds_per_cu=*/65536,
                            /*lds_granule=*/512);
  }
  if (minor == 5) {  // gfx950, MI355
    // AMDGPU.td:1904 gives 160 KiB, and getLdsDwGranularity returns 320
    // dwords rather than gfx942's 128.
    return MakeGfx90aFamily("gfx95x", /*lds_per_cu=*/163840,
                            /*lds_granule=*/1280);
  }

  // Unrecognised gfx9 -- gfx900/906/908, or a part newer than this table.
  // Degrade to LLVM's own pre-GFX90A fallthrough values rather than to
  // anything invented here, so a reader can diff this against AMDGPUBaseInfo
  // and see where it came from. Note these do NOT err in a single direction:
  // max_waves_per_simd goes up (8 -> 10) while total_vgprs goes down
  // (512 -> 256), so which way the answer moves depends on the kernel's
  // register pressure.
  //
  // `exact=false` suppresses the collector's agent-vs-table cross-check, which
  // would otherwise warn on every launch about constants we already know are a
  // guess. That means the cross-check -- the one mechanism that could catch a
  // bad guess -- is off for exactly the targets that are guessing, so the
  // warning below is the only signal a user gets. It has to be here rather
  // than at the emit site: OccupancyStats deliberately carries no `exact`
  // field, because a per-XEvent flag would have to be plumbed through the
  // XStat schema for both vendors.
  LOG_FIRST_N(WARNING, 1)
      << "No validated occupancy constants for gfx_target_version " << v
      << "; falling back to LLVM's pre-GFX90A defaults. Reported occupancy for "
         "this device is an estimate and may be wrong in either direction.";
  AmdGpuTargetConstants g = MakeGfx90aFamily(
      "gfx9-generic", /*lds_per_cu=*/65536, /*lds_granule=*/512);
  g.max_waves_per_simd = 10;   // getMaxWavesPerEU fallthrough
  g.total_vgprs = 256;         // getTotalNumVGPRs fallthrough
  g.vgpr_granule = 4;          // getVGPRAllocGranule fallthrough (wave64)
  g.unified_vgpr_file = false;
  g.exact = false;
  return g;
}

// AMDGPUBaseInfo.cpp getTotalNumVGPRs(has90AInsts, AGPR, VGPR):
//   has90AInsts && AGPR ? alignTo(VGPR, 4) + AGPR : max(VGPR, AGPR)
//
// The max() branch is load-bearing, not defensive. On gfx908 the SDK's
// accum_vgpr_count() *returns* arch_vgpr_count(), so an unconditional sum
// would report MI100 register pressure at exactly 2x. Every caller that needs
// a register count -- including ToXStat's `regs:` token -- must route through
// here rather than adding the two fields itself.
uint32_t UnifiedVgprCount(const AmdGpuTargetConstants& tc, uint32_t arch,
                          uint32_t accum) {
  if (tc.unified_vgpr_file && accum != 0) {
    return SaturatingAdd(AlignTo(arch, tc.arch_vgpr_granule), accum);
  }
  return std::max(arch, accum);
}

std::optional<OccupancyStats> GetOccupancy(const RocmDeviceOccupancyParams& p,
                                           uint32_t cu_count) {
  std::optional<AmdGpuTargetConstants> tc_opt =
      LookupTargetConstants(p.gfx_target_version);
  if (!tc_opt.has_value()) return std::nullopt;
  const AmdGpuTargetConstants& tc = *tc_opt;

  if (p.block_size == 0) return std::nullopt;
  // No register data at all means the code-object symbol lookup missed. We
  // cannot distinguish "a kernel that uses zero registers" (which does not
  // exist) from "we never saw the symbol", so model nothing.
  if (p.arch_vgpr_count == 0 && p.accum_vgpr_count == 0) return std::nullopt;

  const uint32_t slots = tc.max_waves_per_simd * tc.simd_per_cu;
  const uint32_t waves_per_wg = CeilDiv(p.block_size, tc.wave_front_size);
  // A workgroup that cannot fit in a CU's wave slots at all is not a launch
  // geometry the hardware would have accepted; treat it as unmodelable.
  if (waves_per_wg == 0 || waves_per_wg > slots) return std::nullopt;

  const uint32_t vgprs =
      UnifiedVgprCount(tc, p.arch_vgpr_count, p.accum_vgpr_count);

  // ROCm version-skew guard. arch/accum are decoded by the *installed* SDK via
  // a string match on the agent name, and an SDK predating a target's support
  // silently falls through to a generic branch. On a unified-file target the
  // hardware allocation is a multiple of the 8-register granule, so a sum that
  // is not one means we are being handed data whose encoding we do not
  // understand.
  //
  // NOTE, this deviates from the design doc, which applies the check to every
  // unified-file kernel. That over-triggers: `accum == 0` takes the max()
  // branch above, leaving `vgprs == arch_vgpr_count`, and arch counts are
  // granulated to 4 rather than 8, so perfectly ordinary low-register kernels
  // (arch=4, arch=100) would have their occupancy suppressed. Restricting the
  // guard to the sum branch keeps it meaningful. The cost is that the one
  // skew case the doc names -- an old SDK reporting accum=0 on gfx950 -- is
  // not detectable here, because it is numerically indistinguishable from a
  // genuine non-MFMA kernel. That case needs a documented minimum ROCm
  // version, not arithmetic.
  if (tc.unified_vgpr_file && p.accum_vgpr_count != 0 && (vgprs % 8) != 0) {
    LOG_FIRST_N(WARNING, 1)
        << "rocprofiler-sdk reported a non-granulated unified VGPR count for "
        << tc.name << " (arch=" << p.arch_vgpr_count
        << ", accum=" << p.accum_vgpr_count
        << "); the SDK may predate this target. Occupancy suppressed.";
    return std::nullopt;
  }

  // --- VGPR bound, waves/SIMD (getNumWavesPerEUWithNumVGPRs).
  const uint32_t occ_vgpr =
      (vgprs < tc.vgpr_granule)
          ? tc.max_waves_per_simd
          : Clamp(tc.total_vgprs / AlignTo(vgprs, tc.vgpr_granule), 1,
                  tc.max_waves_per_simd);

  // --- SGPR bound, waves/SIMD (getOccupancyWithNumSGPRs, closed form).
  // Live on gfx9: isSGPROccupancyLimited is IsaVersion.Major < 10, and
  // gfx90a/942/950 are all Major 9. It costs at most one wave -- addressable
  // SGPRs is 102, so alignTo(102,16)=112 is the worst case -- but that is
  // 12.5 pp and it is real.
  uint32_t occ_sgpr = tc.max_waves_per_simd;
  if (tc.sgpr_limited && p.sgpr_count != 0) {
    const uint32_t per_wave =
        AlignTo(p.sgpr_count, tc.sgpr_granule) + tc.sgpr_trap_reserve;
    if (per_wave != 0) {
      occ_sgpr = Clamp(tc.total_sgprs / per_wave, 1, tc.max_waves_per_simd);
    }
  }

  // --- Convert the register bounds to whole workgroups per CU.
  //
  // DELIBERATE DIVERGENCE FROM LLVM #1. GCNSubtarget::computeOccupancy takes a
  // flat min() in waves/EU and never re-quantizes to whole workgroups. We do,
  // because hardware cannot resident a partial workgroup and neither can
  // hipOccupancyMaxActiveBlocksPerMultiprocessor. Consequence for anyone
  // adding tests: llc's "; Occupancy:" comment is NOT a valid oracle for this
  // term.
  const uint32_t wgs_vgpr = (occ_vgpr * tc.simd_per_cu) / waves_per_wg;
  const uint32_t wgs_sgpr = (occ_sgpr * tc.simd_per_cu) / waves_per_wg;

  // --- LDS bound, workgroups per CU (AMDGPUSubtarget.cpp).
  uint32_t wgs_lds = kUnbounded;
  if (p.smem_bytes != 0) {
    const uint32_t lds_per_wg = AlignTo(p.smem_bytes, tc.lds_granule);
    wgs_lds = tc.lds_per_cu / std::max(lds_per_wg, 1u);
    // LLVM: a queried LDS size may be larger than a CU has, "in which case we
    // consider the only achievable occupancy to be 1". Returning nothing here
    // would be a silent hole for exactly the kernels most worth flagging.
    if (wgs_lds == 0) wgs_lds = 1;
  }

  // --- Wave-slot and barrier bounds, workgroups per CU.
  const uint32_t wgs_slots = slots / waves_per_wg;
  const uint32_t wgs_barrier =
      (waves_per_wg == 1) ? slots : std::min(wgs_slots, tc.max_barriers);

  const uint32_t wgs = std::max(
      1u, std::min({wgs_vgpr, wgs_sgpr, wgs_lds, wgs_slots, wgs_barrier}));

  // --- Attribute the limiter from the UNQUANTIZED bounds.
  //
  // The obvious version -- compare each quantized bound against the final wgs
  // and let the last match win -- misattributes the pure-quantization cases.
  // For a 320-thread block occ_vgpr is 8 (no VGPR pressure whatsoever) yet
  // wgs_vgpr is 8*4/5 = 6, which ties the answer, so VGPR would claim the
  // blame for a kernel using four registers. Requiring a resource to be
  // strictly below the slot ceiling in its own units before it can claim
  // attribution is what makes wg320 and wg768 correctly report "workgroup" --
  // which is the entire reason the limiter exists.
  OccupancyLimiter limiter = OccupancyLimiter::kNone;
  uint32_t best = wgs_slots;
  if (wgs_barrier < wgs_slots && wgs_barrier <= best) {
    best = wgs_barrier;
    limiter = OccupancyLimiter::kBarrier;
  }
  if (occ_sgpr < tc.max_waves_per_simd && wgs_sgpr <= best) {
    best = wgs_sgpr;
    limiter = OccupancyLimiter::kSGPR;
  }
  if (wgs_lds < wgs_slots && wgs_lds <= best) {
    best = wgs_lds;
    limiter = OccupancyLimiter::kLDS;
  }
  if (occ_vgpr < tc.max_waves_per_simd && wgs_vgpr <= best) {
    best = wgs_vgpr;
    limiter = OccupancyLimiter::kVGPR;
  }
  if (limiter == OccupancyLimiter::kNone && wgs * waves_per_wg < slots) {
    limiter = OccupancyLimiter::kWorkgroup;
  }
  if (wgs * waves_per_wg >= slots) limiter = OccupancyLimiter::kNone;

  OccupancyStats s;
  s.active_blocks_per_cu = wgs;
  s.active_waves_per_cu = std::min(wgs * waves_per_wg, slots);
  s.waves_per_simd =
      static_cast<double>(s.active_waves_per_cu) / tc.simd_per_cu;
  // CUPTI parity: percent of THREADS per CU, not waves.
  //
  // DELIBERATE DIVERGENCE FROM LLVM #2 lives one level up, in what we do NOT
  // copy: AMDGPUSubtarget::getOccupancyWithWorkGroupSizes returns
  // clamp(divideCeil(wavesPerCU, EUsPerCU), 1, Wmax) and the asm printer takes
  // the .second of that range, which rounds 30 waves/CU up to 8/8 = 100%. A
  // profiler knows the exact launch geometry and must report the exact answer.
  s.occupancy_pct = 100.0 * static_cast<double>(wgs) * p.block_size /
                    (static_cast<double>(slots) * tc.wave_front_size);
  s.limiter = limiter;
  s.total_vgprs = vgprs;
  // CUDA's min_grid_size is whole-device, not per-CU.
  s.min_grid_size = cu_count == 0 ? 0 : wgs * cu_count;
  return s;
}

}  // namespace profiler
}  // namespace xla
