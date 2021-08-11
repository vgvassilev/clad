// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad \
// RUN:-Xclang -help %s 2>&1 | FileCheck --check-prefix=CHECK_HELP %s
// CHECK_HELP: -fdump-source-fn
// CHECK_HELP-NEXT: -fdump-source-fn-ast
// CHECK_HELP-NEXT: -fdump-derived-fn
// CHECK_HELP-NEXT: -fdump-derived-fn-ast
// CHECK_HELP-NEXT: -fgenerate-source-file
// CHECK_HELP-NEXT: -fcustom-estimation-model
// CHECK_HELP-NEXT: -help

// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad\
// RUN: -Xclang -invalid %s 2>&1 | FileCheck --check-prefix=CHECK_INVALID %s
// CHECK_INVALID: -invalid

