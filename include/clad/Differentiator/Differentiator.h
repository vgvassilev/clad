//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_DIFFERENTIATOR
#define CLAD_DIFFERENTIATOR

#include "BuiltinDerivatives.h"
#include "FunctionTraits.h"
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

  // for executing non-member functions
  template<class F, class... Args> 
  return_type_t<F> execute_helper(F f, Args&&... args) {
      return f(static_cast<Args>(args)... );
  }

  // for executing member-functions
  template<class ReturnType, class C, class Obj, class... Args> 
  auto execute_helper( ReturnType C::* f, Obj&& obj, Args&&... args) -> return_type_t<decltype(f)>  {
      return (static_cast<Obj>(obj).*f)(static_cast<Args>(args)... );
  }

  // Using std::function and std::mem_fn introduces a lot of overhead, which we
  // do not need. Another disadvantage is that it is difficult to distinguish a
  // 'normal' use of std::{function,mem_fn} from the ones we must differentiate.
  template <typename F> class CladFunction {
  public:
    using CladFunctionType = F;

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
        printf("Make sure calls to clad are within a #pragma clad ON region\n");

        // Invalidate the placeholders.
        m_Function = nullptr;
        m_Code = nullptr;
      }
    }

    // Intentionally leak m_Code, otherwise we have to link against c++ runtime,
    // i.e -lstdc++.
    //~CladFunction() { /*free(m_Code);*/ }

    CladFunctionType getFunctionPtr() { return m_Function; }


    template <typename... Args> return_type_t<F> execute(Args &&... args) {
      if (!m_Function) {
        printf("CladFunction is invalid\n");
        return static_cast<return_type_t<F>>(0);
      }

      return execute_helper(m_Function,static_cast<Args>(args)...);
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
  template<unsigned N = 1, typename ArgSpec = const char *, typename F>
  CladFunction <F> __attribute__((annotate("D")))
  differentiate(F f, ArgSpec args = "", const char* code = "") {
    assert(f && "Must pass in a non-0 argument");
    return CladFunction<F>(f, code);
  }

  /// A function for gradient computation.
  /// Given a function f, clad::gradient generates its gradient f_grad and
  /// returns a CladFunction for it.
  template<typename ArgSpec = const char *, typename F>
  CladFunction<ExtractDerivedFnTraits_t<F>> __attribute__((annotate("G")))
  gradient(F f, ArgSpec args = "", const char* code = "") {
    assert(f && "Must pass in a non-0 argument");
    return CladFunction<ExtractDerivedFnTraits_t<F>>(
      reinterpret_cast<ExtractDerivedFnTraits_t<F>>(f) /* will be replaced by gradient*/,
      code);
  }

  /// Function for Hessian matrix computation
  /// Given  a function f, clad::hessian generates all the second derivatives
  /// of the original function, (they are also columns of a Hessian matrix)
  template<typename ArgSpec = const char *, typename F>
  CladFunction<ExtractDerivedFnTraits_t<F>> __attribute__((annotate("H")))
  hessian(F f, ArgSpec args = "", const char* code = "") {
    assert(f && "Must pass in a non-0 argument");
    return CladFunction<ExtractDerivedFnTraits_t<F>>(
      reinterpret_cast<ExtractDerivedFnTraits_t<F>>(f) /* will be replaced by Hessian*/,
      code);
  }

  template<typename ArgSpec = const char *, typename F>
  CladFunction<ExtractDerivedFnTraits_t<F>> __attribute__((annotate("J")))
  jacobian(F f, ArgSpec args = "", const char* code = "") {
    assert(f && "Must pass in a non-0 argument");
    return CladFunction<ExtractDerivedFnTraits_t<F>>(
        reinterpret_cast<ExtractDerivedFnTraits_t<F>>(f) /* will be replaced by Jacobian*/,
        code);
  }
}
#endif // CLAD_DIFFERENTIATOR

// Enable clad after the header was included.
// FIXME: The header inclusion should be made automatic if the pragma is seen.
#pragma clad ON
