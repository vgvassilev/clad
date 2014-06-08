//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_DIFFERENTIATOR
#define CLAD_DIFFERENTIATOR
// We might want to consider using one of C++11 features of std. For now I am 
// sceptical, because they enforce extra conventions that we don't need. Moreover
// by 1.05.2013 it seems that they are not supported on MacOS.

//#include <functional>
//#include <utility>
//void diff(std::function f) {
//void diff(std::mem_fn f) {

#include "BuiltinDerivatives.h"

extern "C" int printf(const char* fmt, ...);

template<typename ReturnResult, typename... ArgsTypes>
class Function {
public:
  using FunctionType = ReturnResult (*)(ArgsTypes...);
private:
  FunctionType m_Function;
  const char m_Code[];
public:
  Function(FunctionType f)
    : m_Function(f), m_Code("aaaa") { }

  template<typename... Args>
  Function differentiate(Args&&... args);

  template<typename... Args>
  ReturnResult execute(Args&&... args) {
    if(!m_Function)
      printf("Function ptr must be set.");
    printf("I was here. m_Function=%p\n", m_Function);
    ReturnResult result = m_Function(args...);
    return result;
  }

  // template<typename... Args>
  // void executeAndPrintResult(Args&&... args) {
  //   execute(args...);
  //   printf("Returns void\n");
  // }

  // template<typename... Args>
  // int executeAndPrintResult(Args&&... args) {
  //   ReturnResult result = execute(args...);
  //   printf("Returns covertible to int %d\n", result);
  //   return result;
  // }

  // template<typename... Args>
  // double executeAndPrintResult(Args&&... args) {
  //   ReturnResult result = execute(args...);
  //   printf("Returns covertible to double %f\n", result);
  //   return result;
  // }

  void dump() const {
    printf("The code is: %s\n", m_Code);
  }
};

// This is the function which will be instantiated with the concrete arguments
// After that our AD library will have all the needed information. For example:
// which is the differentiated function, which is the argument with respect to.
//
// This will be useful in fucture when we are ready to support partial diff.
//

template<typename F, typename... Args, typename... A>
Function<F, Args...> diff(F (*f)(Args...), A&&... a)
   __attribute__((annotate("D"))) {
  return Function<F, Args...>(f);
}

template<typename F, class C, typename... Args, typename... A>
Function<F, Args...> diff(F (C::*f)(Args...), A&&... a)
   __attribute__((annotate("D"))) {
//return Function<F, Args...>(f);
return 0;
}
#endif // CLAD_DIFFERENTIATOR
