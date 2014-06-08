//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_DIFFERENTIATOR
#define CLAD_DIFFERENTIATOR

#include "BuiltinDerivatives.h"

extern "C" int printf(const char* fmt, ...);

// Using std::function and std::mem_fn introduces a lot of overhead, which we
// do not need. Another disadvantage is that it is difficult to distinguish a
// 'normal' use of std::{function,mem_fn} from the ones we must differentiate.
template<typename ReturnResult, typename... ArgsTypes>
class CladFunction {
public:
  using CladFunctionType = ReturnResult (*)(ArgsTypes...);
private:
  CladFunctionType m_Function;
  const char m_Code[];
public:
  CladFunction(CladFunctionType f)
    : m_Function(f), m_Code("aaaa") { }

  ///\brief N is the derivative order.
  ///
  template<unsigned N>
  CladFunction differentiate(unsigned independentArg);

  template<typename... Args>
  ReturnResult execute(Args&&... args) {
    if(!m_Function)
      printf("Function ptr must be set.");
    printf("I was here. m_Function=%p\n", m_Function);
    ReturnResult result = m_Function(args...);
    return result;
  }

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

template<typename F, typename... Args>
CladFunction<F, Args...> diff(F (*f)(Args...), unsigned independentArg)
   __attribute__((annotate("D"))) {
  return CladFunction<F, Args...>(f);
}

template<typename F, class C, typename... Args>
CladFunction<F, Args...> diff(F (C::*f)(Args...), unsigned independentArg)
   __attribute__((annotate("D"))) {
//return CladFunction<F, Args...>(f);
return 0;
}
#endif // CLAD_DIFFERENTIATOR
