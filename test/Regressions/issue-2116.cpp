
// RUN: %clang++ -fsyntax-only -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang %clad_plugin_path %s 2>&1 | FileCheck %s

// CHECK-NOT: Assertion failed

#pragma clad ON
#pragma clad OFF
#pragma clad OFF

int main() {
    return 0;
}
