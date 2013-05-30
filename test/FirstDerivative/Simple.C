// RUN: %autodiff %s -x c++ -fsyntax-only -ast-dump 2&>1 | FileCheck %s
extern "C" int printf(const char* fmt, ...);

void f() {
  printf("aaa\n");
}

int main() {
  f();
  return 0;
}

//CHECK:aaa
