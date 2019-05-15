//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_DIFFERENTIATOR
#define CLAD_DIFFERENTIATOR

#include "BuiltinDerivatives.h"
#include "Tape.h"

#include <assert.h>
#include <stddef.h>

extern "C" {
  int printf(const char* fmt, ...);
  char* strcpy (char* destination, const char* source);
  size_t strlen(const char*);
#if defined(__APPLE__) || defined(_MSC_VER)
  void* malloc(size_t);
  void free(void *ptr);
#else
  void* malloc(size_t) __THROW __attribute_malloc__ __wur;
  void free(void *ptr) __THROW;
#endif
}

namespace clad {
  /// Tape type used for storing values in reverse-mode AD inside loops.
  template <typename T>
  using tape = tape_impl<T>;

  /// Add value to the end of the tape, return the same value.
  template <typename T>
  T push(tape<T>& to, T val) {
    to.emplace_back(val);
    return val;
  }

  /// Remove the last value from the tape, return it.
  template <typename T>
  T pop(tape<T>& to) {
    T val = to.back();
    to.pop_back();
    return val;
  }

  /// Access return the last value in the tape.
  template <typename T>
  T& back(tape<T>& of) {
    return of.back();
  }

  // Provide the accurate type for standalone functions and members.
  template<bool isMemFn, typename ReturnResult, typename... ArgTypes>
  struct FnTypeTrait {
    using type = ReturnResult (*)(ArgTypes...);

    static ReturnResult execute(type f, ArgTypes&&... args) {
      return f(args...);
    }
  };

  // If it is a member function the compiler uses this specialisation.
  template<typename ReturnResult, class C, typename... ArgTypes>
  struct FnTypeTrait<true, ReturnResult, C, ArgTypes...> {
    using type = ReturnResult (C::*)(ArgTypes...);

    static ReturnResult execute(type f, const C& self, ArgTypes&&... args) {
      // cast away the const and call.
      return (const_cast<C&>(self).*f)(args...);
    }
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
    CladFunction(CladFunctionType f, const char* code) {
      assert(f && "Must pass a non-0 argument.");
      if (size_t length = strlen(code)) {
        m_Function = f;
        m_Code = (char*)malloc(length + 1);
        strcpy(m_Code, code);
      } else {
        // clad did not place the derivative in this object. This can happen
        // upon error of if clad was disabled. Diagnose.
        printf("clad failed to place the generated derivative in the object\n");

        // Invalidate the placeholders.
        m_Function = nullptr;
        m_Code = nullptr;
      }
    }

    // Intentionally leak m_Code, otherwise we have to link against c++ runtime,
    // i.e -lstdc++.
    //~CladFunction() { /*free(m_Code);*/ }

    CladFunctionType getFunctionPtr() { return m_Function; }

    template<typename ...Args>
    ReturnResult execute(Args&&... args) {
      if (!m_Function) {
        printf("CladFunction is invalid\n");
        return static_cast<ReturnResult>(0);
      }
      return FnTypeTrait<isMemFn, ReturnResult, ArgTypes...>
        // static_cast == std::forward, i.e convert the Args to ArgTypes
        ::execute(m_Function, static_cast<ArgTypes>(args)...);
    }

    void dump() const {
      if (m_Code)
        printf("The code is: %s\n", m_Code);
      else
        printf("<invalid>\n");
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
  template<unsigned N = 1, typename ArgSpec = const char *, typename R, typename... Args>
  CladFunction <false, R, Args...> __attribute__((annotate("D")))
  differentiate(R (*f)(Args...), ArgSpec args = "", const char* code = "") {
    assert(f && "Must pass in a non-0 argument");
    return CladFunction<false, R, Args...>(f, code);
  }

  template<unsigned N = 1, typename ArgSpec = const char *, typename R, class C, typename... Args>
  CladFunction<true, R, C, Args...> __attribute__((annotate("D")))
  differentiate(R (C::*f)(Args...), ArgSpec args = "", const char* code = "") {
    assert(f && "Must pass in a non-0 argument");
    return CladFunction<true, R, C, Args...>(f, code);
  }

  /// A function for gradient computation.
  /// Given a function f, clad::gradient generates its gradient f_grad and
  /// returns a CladFunction for it.
  template<typename ArgSpec = const char *, typename R, typename... Args>
  CladFunction<false, void, Args..., R*> __attribute__((annotate("G")))
  gradient(R (*f)(Args...), ArgSpec args = "", const char* code = "") {
    assert(f && "Must pass in a non-0 argument");
    return CladFunction<false, void, Args..., R*>(
      reinterpret_cast<void (*) (Args..., R*)>(f) /* will be replaced by gradient*/,
      code);
  }

  template<typename ArgSpec = const char *, typename R, typename C, typename... Args>
  CladFunction<true, void, C, Args..., R*> __attribute__((annotate("G")))
  gradient(R (C::*f)(Args...), ArgSpec args = "", const char* code = "") {
    assert(f && "Must pass in a non-0 argument");
    return CladFunction<true, void, C, Args..., R*>(
      reinterpret_cast<void (C::*) (Args..., R*)>(f) /* will be replaced by gradient*/,
      code);
  }
}
#endif // CLAD_DIFFERENTIATOR
