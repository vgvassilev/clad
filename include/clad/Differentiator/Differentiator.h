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
  /// \returns the size of a c-style string
  CUDA_HOST_DEVICE unsigned int GetLength(const char* code) {
    unsigned int count;
    const char* code_copy = code;
    #ifdef __CUDACC__
      count = 0;
      while (*code_copy != '\0') {
        count++;
        code_copy++;
      }
    #else
      count = strlen(code_copy);
    #endif
    return count;
  }
  
  /// Tape type used for storing values in reverse-mode AD inside loops.
  template <typename T>
  using tape = tape_impl<T>;

  /// Add value to the end of the tape, return the same value.
  template <typename T>
  CUDA_HOST_DEVICE T push(tape<T>& to, T val) {
    to.emplace_back(val);
    return val;
  }

  /// Remove the last value from the tape, return it.
  template <typename T>
  CUDA_HOST_DEVICE T pop(tape<T>& to) {
    T val = to.back();
    to.pop_back();
    return val;
  }

  /// Access return the last value in the tape.
  template <typename T>
  CUDA_HOST_DEVICE T& back(tape<T>& of) {
    return of.back();
  }

  /// Helper function for executing non-member derived functions.
  /// RedundantType is only present to keep signature same as of execute_helper
  /// member function counterpart.
  template<class F, class RedundantType, class... Args> 
  return_type_t<F> execute_helper(F f, RedundantType* redundant, Args&&... args) {
      return f(static_cast<Args>(args)... );
  }

  /// Helper functions for executing member derived functions.
  /// If user have passed object explicitly, then this specialization will be used and 
  /// derived fn will be called through the passed object.
  template <class ReturnType,
            class C,
            class FunctorType,
            class Obj,
            class = typename std::enable_if<
                std::is_same<typename std::remove_cv<FunctorType>::type,
                             C>::value>::type,
            class = typename std::enable_if<
                std::is_same<typename std::decay<Obj>::type, C>::value>::type,
            class... Args>
  auto execute_helper(ReturnType C::*f,
                      FunctorType* functor,
                      Obj&& obj,
                      Args&&... args) -> return_type_t<decltype(f)> {
    return (static_cast<Obj>(obj).*f)(static_cast<Args>(args)...);
  }

  // If user have not passed object explicitly, then this specialization will be used
  // and derived fn will be called through the object saved in `CladFunction`.
  template <class ReturnType,
            class C,
            class FunctorType,
            class = typename std::enable_if<
                std::is_same<typename std::remove_cv<FunctorType>::type,
                             C>::value>::type,
            class... Args>
  auto execute_helper(ReturnType C::*f, FunctorType* functor, Args&&... args)
      -> return_type_t<decltype(f)> {
    return (functor->*f)(static_cast<Args>(args)...);
  }

  // Using std::function and std::mem_fn introduces a lot of overhead, which we
  // do not need. Another disadvantage is that it is difficult to distinguish a
  // 'normal' use of std::{function,mem_fn} from the ones we must differentiate.
  /// Explicitly passing `FunctorT` type is necessary for maintaining 
  /// const correctness of functor types.
  /// Default value of `Functor` here is temporary, and should be removed once all
  /// clad differentiation functions support differentiating functors.
  template <typename F, typename FunctorT = ExtractFunctorTraits_t<F>> class CladFunction {
  public:
    using CladFunctionType = F;
    using FunctorType = FunctorT;

  private:
    CladFunctionType m_Function;
    char* m_Code;
    FunctorType *m_Functor = nullptr;

  public:
    CUDA_HOST_DEVICE CladFunction(CladFunctionType f,
                                  const char* code,
                                  FunctorType* functor = nullptr)
        : m_Functor(functor) {
      assert(f && "Must pass a non-0 argument.");
      if (size_t length = GetLength(code)) {
        m_Function = f;
        char* temp = (char*)malloc(length + 1);
        m_Code = temp;
        while ((*temp++ = *code++));
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
    /// Constructor overload for initializing `m_Functor` when functor
    /// is passed by reference.
    // ! Please confirm if `CUDA_HOST_DEVICE` should be added in the constructor overload.
    CUDA_HOST_DEVICE
    CladFunction(CladFunctionType f, const char* code, FunctorType& functor)
        : CladFunction(f, code, &functor){};

    // Intentionally leak m_Code, otherwise we have to link against c++ runtime,
    // i.e -lstdc++.
    //~CladFunction() { /*free(m_Code);*/ }

    CladFunctionType getFunctionPtr() { return m_Function; }


    template <typename... Args> return_type_t<F> execute(Args &&... args) {
      if (!m_Function) {
        printf("CladFunction is invalid\n");
        return static_cast<return_type_t<F>>(0);
      }
      /// m_Functor is passed for both member and non-member functions.
      /// m_Functor is ignored when derived function is a free function or when
      /// user have explicitly passed object through which derived function 
      /// should be called.
      return execute_helper(m_Function, m_Functor, static_cast<Args>(args)...);
    }

    /// Return the string representation for the generated derivative.
    const char* getCode() const {
      if (m_Code)
        return m_Code;
      else
        return "<invalid>";
    }
 
    void dump() const {
      printf("The code is: %s\n", getCode());
    }
  };

  // This is the function which will be instantiated with the concrete arguments
  // After that our AD library will have all the needed information. For eg:
  // which is the differentiated function, which is the argument with respect to.
  //
  // This will be useful in fucture when we are ready to support partial diff.
  //

  /// \brief Differentiates function using forward mode.
  ///
  /// Performs partial differentiation of the `fn` argument using forward mode wrt
  /// parameter specified in `args`. To differentiate `fn` wrt several parameters, 
  /// please see `clad::gradient`.
  /// \param[in] fn function to differentiate
  /// \param[in] args independent parameter information
  /// \return `CladFunction` object to access the corresponding derived function.
  template <unsigned N = 1,
            typename ArgSpec = const char*,
            typename F,
            typename DerivedFnType = ExtractDerivedFnTraitsForwMode_t<F>,
            typename = typename std::enable_if<
                !std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>> __attribute__((annotate("D")))
  differentiate(F fn,
                ArgSpec args = "",
                DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
                const char* code = "") {
    assert(fn && "Must pass in a non-0 argument");
    return CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>>(derivedFn, code);
  }

  /// Specialization for differentiating functors.
  /// Specialization is needed because objects have to be passed
  /// by reference whereas functions have to be passed by value.
  template <unsigned N = 1,
            typename ArgSpec = const char*,
            typename F,
            typename DerivedFnType = ExtractDerivedFnTraitsForwMode_t<F>,
            typename = typename std::enable_if<
                std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>> __attribute__((annotate("D")))
  differentiate(F&& f,
                ArgSpec args = "",
                DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
                const char* code = "") {
    return CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>>(derivedFn, code, f);
  }

  /// A function for gradient computation.
  /// Given a function f, clad::gradient generates its gradient f_grad and
  /// returns a CladFunction for it.
  template <typename ArgSpec = const char*,
            typename F,
            typename DerivedFnType = ExtractDerivedFnTraits_t<F>>
  CladFunction<DerivedFnType> __attribute__((annotate("G"))) CUDA_HOST_DEVICE
  gradient(F f,
           ArgSpec args = "",
           DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
           const char* code = "") {
    assert(f && "Must pass in a non-0 argument");
    return CladFunction<DerivedFnType>(derivedFn /* will be replaced by gradient*/,
                                       code);
  }

  /// Function for Hessian matrix computation
  /// Given  a function f, clad::hessian generates all the second derivatives
  /// of the original function, (they are also columns of a Hessian matrix)
  template <typename ArgSpec = const char*,
            typename F,
            typename DerivedFnType = ExtractDerivedFnTraits_t<F>>
  CladFunction<DerivedFnType> __attribute__((annotate("H")))
  hessian(F f,
          ArgSpec args = "",
          DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
          const char* code = "") {
    assert(f && "Must pass in a non-0 argument");
    return CladFunction<DerivedFnType>(derivedFn /* will be replaced by hessian*/,
                                       code);
  }

  template <typename ArgSpec = const char*,
            typename F,
            typename DerivedFnType = ExtractDerivedFnTraits_t<F>>
  CladFunction<DerivedFnType> __attribute__((annotate("J")))
  jacobian(F f,
           ArgSpec args = "",
           DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
           const char* code = "") {
    assert(f && "Must pass in a non-0 argument");
    return CladFunction<DerivedFnType>(
        derivedFn /* will be replaced by Jacobian*/, code);
  }
}
#endif // CLAD_DIFFERENTIATOR

// Enable clad after the header was included.
// FIXME: The header inclusion should be made automatic if the pragma is seen.
#pragma clad ON
