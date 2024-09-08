// RUN: %cladclang %s -I%S/../../include -oClangConsumers.out                  \
// RUN:               -fms-compatibility -DMS_COMPAT -std=c++14 -fmodules      \
// RUN:                -Xclang -print-stats 2>&1 | %filecheck %s
// CHECK-NOT: {{.*error|warning|note:.*}}
//
// RUN: clang -xc -Xclang -add-plugin -Xclang clad -Xclang -load               \
// RUN:       -Xclang %cladlib %s -I%S/../../include -oClangConsumers.out      \
// RUN:       -Xclang -debug-info-kind=limited -Xclang -triple -Xclang bpf-linux-gnu \
// RUN:       -S -emit-llvm -Xclang -target-cpu -Xclang generic                \
// RUN:              -Xclang -print-stats 2>&1 |                               \
// RUN:                  FileCheck -check-prefix=CHECK_C %s
// CHECK_C-NOT: {{.*error|warning|note:.*}}
// XFAIL: clang-7, clang-8, clang-9, target={{i586.*}}, target=arm64-apple-{{.*}}
//
// RUN: clang -xobjective-c -Xclang -add-plugin -Xclang clad -Xclang -load     \
// RUN:       -Xclang %cladlib %s -I%S/../../include -oClangConsumers.out      \
// RUN:              -Xclang -print-stats 2>&1 |                               \
// RUN:                  FileCheck -check-prefix=CHECK_OBJC %s
// CHECK_OBJC-NOT: {{.*error|warning|note:.*}}

#ifdef __cplusplus

#pragma clang module build N
  module N {}
  #pragma clang module contents
    #pragma clang module begin N
      struct f {  void operator()() const {} };
      template <typename T> auto vtemplate = f{};
    #pragma clang module end
#pragma clang module endbuild

#pragma clang module import N

#ifdef MS_COMPAT
class __single_inheritance IncSingle;
#endif // MS_COMPAT

struct V { virtual int f(); };
int V::f() { return 1; }
template <typename T> T f() { return T(); }
int i = f<int>();

// Check if shouldSkipFunctionBody is called.
// RUN: %cladclang -I%S/../../include -fsyntax-only -fmodules \
// RUN:            -Xclang -code-completion-at=%s:%(line-1):1 %s -o - | \
// RUN:               FileCheck -check-prefix=CHECK-CODECOMP %s
// CHECK-CODECOMP: COMPLETION

// CHECK: HandleImplicitImportDecl
// CHECK: AssignInheritanceModel
// CHECK: HandleTopLevelDecl
// CHECK: HandleCXXImplicitFunctionInstantiation
// CHECK: HandleInterestingDecl
// CHECK: HandleVTable
// CHECK: HandleCXXStaticMemberVarInstantiation

#endif // __cplusplus

#ifdef __STDC_VERSION__ // C mode
int i;

extern char ch;
int test(void) { return ch; }
char ch = 1;

// CHECK_C: CompleteTentativeDefinition
// CHECK_C: CompleteExternalDeclaration
#endif // __STDC_VERSION__

#ifdef __OBJC__
@interface I
void f();
@end
// CHECK_OBJC: HandleTopLevelDeclInObjCContainer
#endif // __OBJC__

int main() {
#ifdef __cplusplus
  vtemplate<int>();
#endif // __cplusplus
}
