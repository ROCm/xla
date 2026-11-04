/* Copyright 2026 The OpenXLA Authors.

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

// Private definitions of the only libstdc++ library functions that
// hipblaslt_ext_shim.cc needs. Everything else it uses from the C++ runtime
// (operator new, __cxa_*, the personality routine, ...) has the same name in
// every C++ runtime, so any build provides it.
//
// With these, the shim links into code built against libc++ without
// libstdc++.so on the link line. That matters: if libstdc++.so is linked into
// such code, the linker resolves its C++ ABI symbols (__dynamic_cast, the
// __cxxabiv1 type_info vtables, ...) against libstdc++ instead of libc++abi,
// and RTTI from the two runtimes no longer matches (dynamic_cast throws
// std::bad_cast).
//
// The definitions are hidden, so they bind the shim's own calls and are never
// exported, also when libstdc++ itself is linked. This file includes no C++
// standard library header on purpose: libstdc++ declares these functions with
// default visibility, which would override the hidden attribute.
//
// They are also weak, because the other runtime may define some of them too:
// libc++ defines std::__throw_bad_alloc() (in libc++.a, next to std::nothrow).
// A strong definition wins over these without a duplicate symbol error, and
// it is just as good: it throws std::bad_alloc, which both runtimes share.
//
// libstdc++'s headers call these only when a std::vector is asked for more
// elements than max_size() (or n * sizeof(T) overflows); running out of memory
// is reported by operator new instead. The shim only sizes vectors by the
// group count, so they cannot run unless that count is corrupt. They report
// the error and abort instead of throwing, which would need libstdc++'s
// exception types.
//
// To check that nothing else is needed, compile both files against libstdc++
// and the hipBLASLt headers of the ROCm in use, and list `nm -u` symbols that
// start with _ZSt, _ZNSt or _ZNKSt: only std::terminate(), which every C++
// runtime provides, may remain.

#include <stdio.h>
#include <stdlib.h>

#define XLA_HIPBLASLT_SHIM_WEAK_HIDDEN \
  __attribute__((weak, visibility("hidden")))

namespace std {

[[noreturn]] XLA_HIPBLASLT_SHIM_WEAK_HIDDEN void __throw_length_error(
    const char* what) {
  fprintf(stderr, "hipblaslt_ext_shim: length error: %s\n", what);
  abort();
}

[[noreturn]] XLA_HIPBLASLT_SHIM_WEAK_HIDDEN void __throw_bad_alloc() {
  fputs("hipblaslt_ext_shim: allocation larger than max_size()\n", stderr);
  abort();
}

// Used by GCC 11+ headers in unoptimized builds.
[[noreturn]] XLA_HIPBLASLT_SHIM_WEAK_HIDDEN void
__throw_bad_array_new_length() {
  fputs("hipblaslt_ext_shim: bad array new length\n", stderr);
  abort();
}

// hipBLASLt's type headers (hipblaslt_e8.h, hipblaslt_e5m3.h) include
// <iostream> for their std::ostream operator<< declarations. Before GCC 13,
// <iostream> also adds a static std::ios_base::Init object to every file that
// includes it, to initialize std::cout and friends. The shim never uses those
// streams, and hipBLASLt's own files still initialize them in its library, so
// the shim's copy does nothing. Only the declarations needed to define it are
// repeated here; libc++'s ios_base is in std::__1, so it does not clash.
class ios_base {
 public:
  class Init {
   public:
    Init();
    ~Init();
  };
};

XLA_HIPBLASLT_SHIM_WEAK_HIDDEN ios_base::Init::Init() {}
XLA_HIPBLASLT_SHIM_WEAK_HIDDEN ios_base::Init::~Init() {}

}  // namespace std
