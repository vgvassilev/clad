//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_DIFFERENTIATOR
#define CLAD_DIFFERENTIATOR

#include "BuiltinDerivatives.h"
#include <assert.h>
#include <stddef.h>

extern "C" {
  int printf(const char* fmt, ...);
  char* strcpy (char* destination, const char* source);
  size_t strlen(const char*);
  void* malloc(size_t) __THROW __attribute_malloc__ __wur;
  void free(void *ptr) __THROW;
}

namespace clad {

  // Provide the accurate type for standalone functions and members.
  template<bool isMemFn, typename ReturnResult, typename... ArgTypes>
  struct FnTypeTrait {
    using type = ReturnResult (*)(ArgTypes...);
  };

  // If it is a member function the compiler uses this specialisation.
  template<typename ReturnResult, class C, typename... ArgTypes>
  struct FnTypeTrait<true, ReturnResult, C, ArgTypes...> {
    using type = ReturnResult (C::*)(ArgTypes...);
  };

  // Using std::function and std::mem_fn introduces a lot of overhead, which we
  // do not need. Another disadvantage is that it is difficult to distinguish a
  // 'normal' use of std::{function,mem_fn} from the ones we must differentiate.
  template<bool isMemFn, typename ReturnResult, typename... ArgTypes>
  class CladFunction {
  public:
    using CladFunctionType = typename FnTypeTrait<isMemFn, ReturnResult, ArgTypes...>::type;
  private:
    CladFunctionType m_Function;
    char* m_Code;
  public:
    CladFunction(CladFunctionType f, const char* code)
      : m_Function(f) {
      assert(f && "Must pass a non-0 argument.");
      m_Code = (char*)malloc(strlen(code) + 1);
      strcpy(m_Code, code);
    }

    // Intentionally leak m_Code, otherwise we have to link against c++ runtime,
    // i.e -lstdc++.
    //~CladFunction() { /*free(m_Code);*/ }

    CladFunctionType getFunctionPtr() { return m_Function; }

    template<typename... Args>
    ReturnResult execute(Args&&... args) {
      ReturnResult result = m_Function(args...);
      return result;
    }

    void dump() const {
      printf("The code is: %s\n", m_Code);
    }
  };

  // This is the function which will be instantiated with the concrete arguments
  // After that our AD library will have all the needed information. For eg:
  // which is the differentiated function, which is the argument with respect to.
  //
  // This will be useful in fucture when we are ready to support partial diff.
  //

  ///\brief N is the derivative order.
  ///
  template<unsigned N = 1, typename R, typename... Args>
  CladFunction <false, R, Args...> __attribute__((annotate("D")))
  differentiate(R (*f)(Args...), unsigned independentArg, const char* code = "") {
    assert(f && "Must pass in a non-0 argument");
    return CladFunction<false, R, Args...>(f, code);
  }

  template<unsigned N = 1, typename R, class C, typename... Args>
  CladFunction<true, R, C, Args...> __attribute__((annotate("D")))
  differentiate(R (C::*f)(Args...), unsigned independentArg, const char* code = "") {
    assert(f && "Must pass in a non-0 argument");
    return CladFunction<true, R, C, Args...>(f, code);
  }
}
#endif // CLAD_DIFFERENTIATOR
