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

// Link-time stand-in for libstdc++.so.6, used when hipblaslt_ext_shim is linked
// into code built against libc++. It exports exactly the libstdc++ symbols that
// hipblaslt_ext_shim.cc references (by version, see XLA_LIBSTDCXX_SYMBOL
// there), so those references resolve at link time, e.g. with --no-undefined.
//
// The symbols are exported as non-default versions (a single "@"), which
// unversioned references cannot bind to: the libc++ code in the same link keeps
// resolving its C++ runtime symbols to libc++, whatever the link order. Putting
// the real libstdc++.so on such a link line would instead bind them to
// libstdc++, mixing the two runtimes (RTTI from both breaks dynamic_cast).
//
// Its SONAME is libstdc++.so.6, so at run time the real libstdc++ is loaded.
// The stub itself is never loaded or distributed.
//
// The versioned symbols inherit the visibility of the stub_* definitions, so
// those must stay default also under -fvisibility=hidden (e.g. JAX's bazelrc).
#define XLA_LIBSTDCXX_STUB_EXPORT __attribute__((visibility("default")))

#define XLA_LIBSTDCXX_STUB(symbol, version)             \
  XLA_LIBSTDCXX_STUB_EXPORT void stub_##symbol(void) {} \
  __asm__(".symver stub_" #symbol ", " #symbol "@" version);

XLA_LIBSTDCXX_STUB(_Znwm, "GLIBCXX_3.4")
XLA_LIBSTDCXX_STUB(_ZdlPv, "GLIBCXX_3.4")
XLA_LIBSTDCXX_STUB(_ZdlPvm, "CXXABI_1.3.9")
XLA_LIBSTDCXX_STUB(_ZNSt8ios_base4InitC1Ev, "GLIBCXX_3.4")
XLA_LIBSTDCXX_STUB(_ZNSt8ios_base4InitD1Ev, "GLIBCXX_3.4")
XLA_LIBSTDCXX_STUB(_ZSt17__throw_bad_allocv, "GLIBCXX_3.4")
XLA_LIBSTDCXX_STUB(_ZSt20__throw_length_errorPKc, "GLIBCXX_3.4")
XLA_LIBSTDCXX_STUB(_ZSt28__throw_bad_array_new_lengthv, "GLIBCXX_3.4.29")

// The one data symbol: _ZTVN10__cxxabiv117__class_type_infoE, a vtable.
XLA_LIBSTDCXX_STUB_EXPORT const void* stub_class_type_info_vtable[3] = {0};
__asm__(
    ".symver stub_class_type_info_vtable, "
    "_ZTVN10__cxxabiv117__class_type_infoE@CXXABI_1.3");
