// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad \
// RUN:-Xclang -help %s 2>&1 | FileCheck --check-prefix=CHECK_HELP %s
// The screen is the option table read out in order, so every option clad
// takes is on it and the header is a line of its own.
// CHECK_HELP: Options specific to Clad (preceded by -plugin-arg-clad):
// CHECK_HELP-NEXT: -Rclad-analysis=<name>
// CHECK_HELP-NEXT: -fclad-porting-hints
// CHECK_HELP-NEXT: -fcustom-estimation-model
// CHECK_HELP-NEXT: -fdisable-analysis=<name>
// CHECK_HELP-NEXT: -fdump-analysis=<name>
// CHECK_HELP-NEXT: -fdump-derived-fn -
// CHECK_HELP-NEXT: -fdump-derived-fn-ast
// CHECK_HELP-NEXT: -fdump-generated-source
// CHECK_HELP-NEXT: -fdump-source-fn -
// CHECK_HELP-NEXT: -fdump-source-fn-ast
// CHECK_HELP-NEXT: -fenable-analysis=<name>
// CHECK_HELP-NEXT: -fgenerate-source-file
// CHECK_HELP-NEXT: -fgenerated-source-dir=<dir>
// CHECK_HELP-NEXT: -fno-validate-clang-version
// CHECK_HELP-NEXT: -fprint-num-diff-errors
// CHECK_HELP-NEXT: -help
// CHECK_HELP-NEXT: -v -
// CHECK_HELP-NEXT: -version - Prints
// Every analysis clad knows about is listed, with the default it runs at.
// CHECK_HELP: -enable-tbr / -disable-tbr {{.*}} Default: on.
// CHECK_HELP-NEXT: -enable-va / -disable-va {{.*}} Default: off.
// CHECK_HELP-NEXT: -enable-ua / -disable-ua {{.*}} Default: off.
// CHECK_HELP-NEXT: -enable-loop / -disable-loop {{.*}} Default: on.

// An unknown option that is nothing like one clad has is reported on its
// own: a suggestion that far out would mislead rather than help.
// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad\
// RUN: -Xclang -invalid %s 2>&1 | FileCheck --check-prefix=CHECK_INVALID %s
// CHECK_INVALID: invalid option -invalid
// CHECK_INVALID-NOT: did you mean

// One edit away, the option it was meant to be is named.
// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad \
// RUN: -Xclang -fdump-derivd-fn %s 2>&1 | FileCheck --check-prefix=CHECK_TYPO %s
// CHECK_TYPO: invalid option -fdump-derivd-fn; did you mean -fdump-derived-fn?

// A joined option is measured by its spelling, not by the value after it,
// which would otherwise swamp the distance.
// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad \
// RUN: -Xclang -Rclad-analysi=loop %s 2>&1 | FileCheck --check-prefix=CHECK_TYPO_EQ %s
// CHECK_TYPO_EQ: invalid option -Rclad-analysi=loop; did you mean -Rclad-analysis=?

// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad\
// RUN: -Xclang -version %s 2>&1 | FileCheck --check-prefix=CHECK_VERSION %s
// CHECK_VERSION: clad version {{[0-9]+\.[0-9]+\.[0-9]+}}

// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad \
// RUN:  -Xclang -fcustom-estimation-model %s 2>&1 | FileCheck --check-prefix=CHECK_EST_INVALID %s
// CHECK_EST_INVALID: `-fcustom-estimation-model` is deprecated

// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad -Xclang -enable-tbr \
// RUN:  -Xclang -plugin-arg-clad -Xclang -disable-tbr %s 2>&1 | FileCheck --check-prefix=CHECK_TBR %s
// CHECK_TBR: -enable-tbr and -disable-tbr cannot be used together

// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad -Xclang -enable-va \
// RUN:  -Xclang -plugin-arg-clad -Xclang -disable-va %s 2>&1 | FileCheck --check-prefix=CHECK_VA %s
// CHECK_VA: -enable-va and -disable-va cannot be used together

// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad -Xclang -enable-ua \
// RUN:  -Xclang -plugin-arg-clad -Xclang -disable-ua %s 2>&1 | FileCheck --check-prefix=CHECK_UA %s
// CHECK_UA: -enable-ua and -disable-ua cannot be used together

// An analysis has to be one clad has; the message names the ones it does.
// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad \
// RUN:  -Xclang -fdisable-analysis=nosuch %s 2>&1 | FileCheck --check-prefix=CHECK_NO_SUCH %s
// CHECK_NO_SUCH: unknown analysis 'nosuch'; known: tbr activity useful

// The same holds for the flag that asks an analysis to report itself.
// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad \
// RUN:  -Xclang -fdump-analysis=nosuch %s 2>&1 | FileCheck --check-prefix=CHECK_NO_DUMP %s
// CHECK_NO_DUMP: unknown analysis 'nosuch'

// 'all' asks for the conservative derivative, so it only makes sense as
// something to turn off.
// RUN: clang -fsyntax-only -fplugin=%cladlib -Xclang -plugin-arg-clad \
// RUN:  -Xclang -fenable-analysis=all %s 2>&1 | FileCheck --check-prefix=CHECK_ENABLE_ALL %s
// CHECK_ENABLE_ALL: unknown analysis 'all'
