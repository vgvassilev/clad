// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad \
// RUN:-Xclang -help %s 2>&1 | FileCheck --check-prefix=CHECK_HELP %s
// CHECK_HELP: -fdump-source-fn
// CHECK_HELP-NEXT: -fdump-source-fn-ast
// CHECK_HELP-NEXT: -fdump-derived-fn
// CHECK_HELP-NEXT: -fdump-derived-fn-ast
// CHECK_HELP-NEXT: -fgenerate-source-file
// CHECK_HELP-NEXT: -fno-validate-clang-version
// CHECK_HELP-NEXT: -enable-tbr
// CHECK_HELP-NEXT: -disable-tbr
// CHECK_HELP-NEXT: -fcustom-estimation-model
// CHECK_HELP-NEXT: -fprint-num-diff-errors
// CHECK_HELP-NEXT: -help

// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad\
// RUN: -Xclang -invalid %s 2>&1 | FileCheck --check-prefix=CHECK_INVALID %s
// CHECK_INVALID: -invalid

// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad\
// RUN: -Xclang -version %s 2>&1 | FileCheck --check-prefix=CHECK_VERSION %s
// CHECK_VERSION: clad version {{[0-9]+\.[0-9]+\.[0-9]+}}

// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad \
// RUN:  -Xclang -fcustom-estimation-model %s 2>&1 | FileCheck --check-prefix=CHECK_EST_INVALID %s
// CHECK_EST_INVALID: No shared object was specified

// RUN: touch %t.so
// RUN: ! %cladclang -fsyntax-only  -Xclang -plugin-arg-clad \
// RUN:  -Xclang -fcustom-estimation-model -Xclang -plugin-arg-clad \
// RUN: -Xclang %t.so %S/../../demos/ErrorEstimation/CustomModel/test.cpp \
// RUN: -I%S/../../include 2>&1 | FileCheck --check-prefix=CHECK_SO_INVALID %s
// CHECK_SO_INVALID: Failed to load '{{.*.so}}', {{.*}}. Aborting.

// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad -Xclang -enable-tbr \
// RUN:  -Xclang -plugin-arg-clad -Xclang -disable-tbr %s 2>&1 | FileCheck --check-prefix=CHECK_TBR %s
// CHECK_TBR: -enable-tbr and -disable-tbr cannot be used together

// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad -Xclang -enable-va \
// RUN:  -Xclang -plugin-arg-clad -Xclang -disable-va %s 2>&1 | FileCheck --check-prefix=CHECK_VA %s
// CHECK_VA: -enable-va and -disable-va cannot be used together

// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad -Xclang -enable-ua \
// RUN:  -Xclang -plugin-arg-clad -Xclang -disable-ua %s 2>&1 | FileCheck --check-prefix=CHECK_UA %s
// CHECK_UA: -enable-ua and -disable-ua cannot be used together
