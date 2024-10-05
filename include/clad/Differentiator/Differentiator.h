//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_DIFFERENTIATOR
#define CLAD_DIFFERENTIATOR

#include "Array.h"
#include "ArrayRef.h"
#include "BuiltinDerivatives.h"
#include "CladConfig.h"
#include "DynamicGraph.h"
#include "FunctionTraits.h"
#include "Matrix.h"
#include "NumericalDiff.h"
#include "Tape.h"

#include <assert.h>
#include <stddef.h>
#include <cstring>

namespace clad {

/// \returns the size of a c-style string
inline CUDA_HOST_DEVICE unsigned int GetLength(const char* code) {
  const char* code_copy = code;
#ifdef __CUDACC__
  unsigned int count = 0;
  while (*code_copy != '\0') {
    count++;
    code_copy++;
  }
#else
  unsigned int count = strlen(code_copy);
#endif
  return count;
}

#ifdef __CUDACC__
#define CUDA_ARGS bool CUDAkernel, dim3 grid, dim3 block,
#define CUDA_REST_ARGS size_t shared_mem, cudaStream_t stream,
#else
#define CUDA_ARGS
#define CUDA_REST_ARGS
#endif

/// Tape type used for storing values in reverse-mode AD inside loops.
template <typename T> using tape = tape_impl<T>;

/// Add value to the end of the tape, return the same value.
template <typename T, typename... ArgsT>
CUDA_HOST_DEVICE T push(tape<T>& to, ArgsT... val) {
  to.emplace_back(std::forward<ArgsT>(val)...);
  return to.back();
}

  /// Add value to the end of the tape, return the same value.
  /// A specialization for clad::array_ref types to use in reverse mode.
  template <typename T, typename U>
  CUDA_HOST_DEVICE clad::array_ref<T> push(tape<clad::array_ref<T>>& to,
                                           U val) {
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
  template <typename T> CUDA_HOST_DEVICE T& back(tape<T>& of) {
    return of.back();
  }

  /// The purpose of this function is to initialize adjoints
  /// (or all of its differentiable fields) with 0.
  // FIXME: Add support for objects.
  /// Initialize a non-array variable.
  template <typename T> CUDA_HOST_DEVICE void zero_init(T& x) { new (&x) T(); }

  /// Initialize a non-const sized array when the size is known and is equal to
  /// N.
  template <typename T> CUDA_HOST_DEVICE void zero_init(T* x, std::size_t N) {
    for (std::size_t i = 0; i < N; ++i)
    zero_init(x[i]);
  }

  /// Initialize a const sized array.
  // NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays)
  template <typename T, std::size_t N>
  CUDA_HOST_DEVICE void zero_init(T (&arr)[N]) {
    zero_init((T*)arr, N);
  }
  // NOLINTEND(cppcoreguidelines-avoid-c-arrays)

  /// Pad the args supplied with nullptr(s) or zeros to match the the num of
  /// params of the function and then execute the function using the padded args
  /// i.e. we are adding default arguments as we cannot do that with
  /// meta programming
  ///
  /// For example:
  /// Let's assume we have a function with the signature:
  ///   fn_grad(double i, double j, int k, int l);
  /// and f is a pointer to fn_grad
  /// and args are the supplied arguments- 1.0, 2.0 and Args has their type
  /// (double, double)
  ///
  /// When pad_and_execute(DropArgs_t<sizeof...(Args), decltype(f)>{}, f, args)
  /// is run, the Rest variadic argument will have the types (int, int).
  /// pad_and_execute will then make up for the remaining args by appending 0s
  /// and the return statement translates to:
  ///   return f(1.0, 2.0, 0, 0);
  // for executing non-member functions
  template <bool EnablePadding, class... Rest, class F, class... Args,
            class... fArgTypes,
            typename std::enable_if<EnablePadding, bool>::type = true>
  CUDA_HOST_DEVICE return_type_t<F>
  execute_with_default_args(list<Rest...>, F f, list<fArgTypes...>,
                            CUDA_ARGS CUDA_REST_ARGS Args&&... args) {
#if defined(__CUDACC__) && !defined(__CUDA_ARCH__)
    if (CUDAkernel) {
      constexpr size_t totalArgs = sizeof...(args) + sizeof...(Rest);
      std::vector<void*> argPtrs;
      argPtrs.reserve(totalArgs);
      (argPtrs.push_back(static_cast<void*>(&args)), ...);

      void* null_param = nullptr;
      for (size_t i = sizeof...(args); i < totalArgs; ++i)
        argPtrs[i] = &null_param;

      cudaLaunchKernel((void*)f, grid, block, argPtrs.data(), shared_mem,
                       stream);
      return return_type_t<F>();
    } else {
      return f(static_cast<Args>(args)..., static_cast<Rest>(nullptr)...);
    }
#else
    return f(static_cast<Args>(args)..., static_cast<Rest>(nullptr)...);
#endif
  }

  template <bool EnablePadding, class... Rest, class F, class... Args,
            class... fArgTypes,
            typename std::enable_if<!EnablePadding, bool>::type = true>
  return_type_t<F>
  execute_with_default_args(list<Rest...>, F f, list<fArgTypes...>,
                            CUDA_ARGS CUDA_REST_ARGS Args&&... args) {
#if defined(__CUDACC__) && !defined(__CUDA_ARCH__)
    if (CUDAkernel) {
      void* argPtrs[] = {(void*)&args...};
      cudaLaunchKernel((void*)f, grid, block, argPtrs, shared_mem, stream);
      return return_type_t<F>();
    }
    return f(static_cast<Args>(args)...);
#else
    return f(static_cast<Args>(args)...);
#endif
  }

  // for executing member-functions
  template <bool EnablePadding, class... Rest, class ReturnType, class C,
            class Obj, class... Args, class... fArgTypes,
            typename std::enable_if<EnablePadding, bool>::type = true>
  CUDA_HOST_DEVICE auto
  execute_with_default_args(list<Rest...>, ReturnType C::*f, Obj&& obj,
                            list<fArgTypes...>, Args&&... args)
      -> return_type_t<decltype(f)> {
    return (static_cast<Obj>(obj).*f)((fArgTypes)(args)...,
                                      static_cast<Rest>(nullptr)...);
  }

  template <bool EnablePadding, class... Rest, class ReturnType, class C,
            class Obj, class... Args, class... fArgTypes,
            typename std::enable_if<!EnablePadding, bool>::type = true>
  auto execute_with_default_args(list<Rest...>, ReturnType C::*f, Obj&& obj,
                                 list<fArgTypes...>, Args&&... args)
      -> return_type_t<decltype(f)> {
    return (static_cast<Obj>(obj).*f)(static_cast<Args>(args)...);
  }

  // Using std::function and std::mem_fn introduces a lot of overhead, which we
  // do not need. Another disadvantage is that it is difficult to distinguish a
  // 'normal' use of std::{function,mem_fn} from the ones we must differentiate.
  /// Explicitly passing `FunctorT` type is necessary for maintaining
  /// const correctness of functor types.
  /// Default value of `Functor` here is temporary, and should be removed
  /// once all clad differentiation functions support differentiating functors.
  template <typename F, typename FunctorT = ExtractFunctorTraits_t<F>,
            bool EnablePadding = false>
  class CladFunction {
  public:
    using CladFunctionType = F;
    using FunctorType = FunctorT;

  private:
    CladFunctionType m_Function;
    char* m_Code;
    FunctorType *m_Functor = nullptr;
    bool m_CUDAkernel = false;

  public:
    CUDA_HOST_DEVICE CladFunction(CladFunctionType f, const char* code,
                                  FunctorType* functor = nullptr,
                                  bool CUDAkernel = false)
        : m_Function(f), m_Functor(functor), m_CUDAkernel(CUDAkernel) {
#ifndef __CLAD_SO_LOADED
      static_assert(false, "clad doesn't appear to be loaded; make sure that "
                           "you pass clad.so to clang.");
#endif

      size_t length = GetLength(code);
      char* temp = (char*)malloc(length + 1);
      m_Code = temp;
      while ((*temp++ = *code++))
        ;
    }
    /// Constructor overload for initializing `m_Functor` when functor
    /// is passed by reference.
    CUDA_HOST_DEVICE
    CladFunction(CladFunctionType f, const char* code, FunctorType& functor)
        : CladFunction(f, code, &functor) {};

    // Intentionally leak m_Code, otherwise we have to link against c++ runtime,
    // i.e -lstdc++.
    //~CladFunction() { /*free(m_Code);*/ }

    CladFunctionType getFunctionPtr() { return m_Function; }

    template <typename... Args, class FnType = CladFunctionType>
    typename std::enable_if<!std::is_same<FnType, NoFunction*>::value,
                            return_type_t<F>>::type
    execute(Args&&... args) CUDA_HOST_DEVICE {
      if (!m_Function) {
        printf("CladFunction is invalid\n");
        return static_cast<return_type_t<F>>(return_type_t<F>());
      }
      if (m_CUDAkernel) {
        printf("Use execute_kernel() for global CUDA kernels\n");
        return static_cast<return_type_t<F>>(return_type_t<F>());
      }
      // here static_cast is used to achieve perfect forwarding
#ifdef __CUDACC__
      return execute_helper(m_Function, m_CUDAkernel, dim3(0), dim3(0),
                            static_cast<Args>(args)...);
#else
      return execute_helper(m_Function, static_cast<Args>(args)...);
#endif
    }

#ifdef __CUDACC__
    template <typename... Args, class FnType = CladFunctionType>
    typename std::enable_if<!std::is_same<FnType, NoFunction*>::value,
                            return_type_t<F>>::type
    execute_kernel(dim3 grid, dim3 block, Args&&... args) CUDA_HOST_DEVICE {
      if (!m_Function) {
        printf("CladFunction is invalid\n");
        return static_cast<return_type_t<F>>(return_type_t<F>());
      }
      if (!m_CUDAkernel) {
        printf("Use execute() for non-global CUDA kernels\n");
        return static_cast<return_type_t<F>>(return_type_t<F>());
      }

      return execute_helper(m_Function, m_CUDAkernel, grid, block,
                            static_cast<Args>(args)...);
    }
#endif

    /// `Execute` overload to be used when derived function type cannot be
    /// deduced. One reason for this can be when user tries to differentiate
    /// an object of class which do not have user-defined call operator.
    /// Error handling is handled in the clad side using clang diagnostics 
    /// subsystem.
    template <typename... Args, class FnType = CladFunctionType>
    typename std::enable_if<std::is_same<FnType, NoFunction*>::value,
                            return_type_t<F>>::type
    execute(Args&&... args) CUDA_HOST_DEVICE {
      return static_cast<return_type_t<F>>(0);
    }

    /// Return the string representation for the generated derivative.
    const char* getCode() const {
      if (m_Code)
        return m_Code;
      else
        return "<invalid>";
    }
    void dump() const {
      printf("The code is: \n%s\n", getCode());
    }

    /// Set object pointed by the functor as the default object for
    /// executing derived member function.
    void setObject(FunctorType* functor) {
      m_Functor = functor;
    } 

    /// Set functor object as the default object for executing derived
    // member function.
    void setObject(FunctorType& functor) {
      m_Functor = &functor;
    }

    /// Clears default object (if any) for executing derived member function.
    void clearObject() {
      m_Functor = nullptr;
    }

    private:
      /// Helper function for executing non-member derived functions.
      template <class Fn, class... Args>
      CUDA_HOST_DEVICE return_type_t<CladFunctionType>
      execute_helper(Fn f, CUDA_ARGS Args&&... args) {
        // `static_cast` is required here for perfect forwarding.
#if defined(__CUDACC__)
        if constexpr (sizeof...(Args) >= 2) {
          auto secondArg =
              std::get<1>(std::forward_as_tuple(std::forward<Args>(args)...));
          if constexpr (std::is_same<std::decay_t<decltype(secondArg)>,
                                     cudaStream_t>::value) {
            return [&](auto shared_mem, cudaStream_t stream, auto&&... args_) {
              return execute_with_default_args<EnablePadding>(
                  DropArgs_t<sizeof...(Args) - 2, F>{}, f,
                  TakeNFirstArgs_t<sizeof...(Args) - 2, decltype(f)>{},
                  CUDAkernel, grid, block, shared_mem, stream,
                  static_cast<decltype(args_)>(args_)...);
            }(static_cast<Args>(args)...);
          } else {
            return execute_with_default_args<EnablePadding>(
                DropArgs_t<sizeof...(Args), F>{}, f,
                TakeNFirstArgs_t<sizeof...(Args), decltype(f)>{}, CUDAkernel,
                grid, block, 0, nullptr, static_cast<Args>(args)...);
          }
        } else {
          return execute_with_default_args<EnablePadding>(
              DropArgs_t<sizeof...(Args), F>{}, f,
              TakeNFirstArgs_t<sizeof...(Args), decltype(f)>{}, CUDAkernel,
              grid, block, 0, nullptr, static_cast<Args>(args)...);
        }
#else
        return execute_with_default_args<EnablePadding>(
            DropArgs_t<sizeof...(Args), F>{}, f,
            TakeNFirstArgs_t<sizeof...(Args), decltype(f)>{},
            static_cast<Args>(args)...);
#endif
      }

      /// Helper functions for executing member derived functions.
      /// If user have passed object explicitly, then this specialization will
      /// be used and derived function will be called through the passed object.
      template <
          class ReturnType,
          class C,
          class Obj,
          class = typename std::enable_if<
              std::is_same<typename std::decay<Obj>::type, C>::value>::type,
          class... Args>
      return_type_t<CladFunctionType>
      execute_helper(ReturnType C::*f, Obj&& obj, Args&&... args) {
        // `static_cast` is required here for perfect forwarding.
      return execute_with_default_args<EnablePadding>(
          DropArgs_t<sizeof...(Args), decltype(f)>{}, f, static_cast<Obj>(obj),
          TakeNFirstArgs_t<sizeof...(Args), decltype(f)>{},
          static_cast<Args>(args)...);
      }
      /// If user have not passed object explicitly, then this specialization
      /// will be used and derived function will be called through the object
      /// saved in `CladFunction`.
      template <class ReturnType, class C, class... Args>
      return_type_t<CladFunctionType> execute_helper(ReturnType C::*f,
                                                     Args&&... args) {
        // `static_cast` is required here for perfect forwarding.
        return execute_with_default_args<EnablePadding>(
            DropArgs_t<sizeof...(Args), decltype(f)>{}, f, *m_Functor,
            TakeNFirstArgs_t<sizeof...(Args), decltype(f)>{},
            static_cast<Args>(args)...);
      }
  };

  // This is the function which will be instantiated with the concrete arguments
  // After that our AD library will have all the needed information. For eg:
  // which is the differentiated function, which is the argument with respect
  // to.
  //
  // This will be useful in future when we are ready to support partial diff.
  //

  /// Differentiates function using forward mode.
  ///
  /// Performs partial differentiation of the `fn` argument using forward mode
  /// wrt parameter specified in `args`. Template parameter `BitMaskedOpts`
  /// denotes the derivative order and any extra options. To differentiate `fn`
  /// wrt several parameters, please see `clad::gradient`. \param[in] fn
  /// function to differentiate \param[in] args independent parameter
  /// information \returns `CladFunction` object to access the corresponding
  /// derived function.
  template <unsigned... BitMaskedOpts, typename ArgSpec = const char*,
            typename F,
            typename DerivedFnType = ExtractDerivedFnTraitsForwMode_t<F>,
            typename = typename std::enable_if<
                !clad::HasOption(GetBitmaskedOpts(BitMaskedOpts...),
                                 opts::vector_mode) &&
                !std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>> __attribute__((
      annotate("D")))
  differentiate(F fn, ArgSpec args = "",
                DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
                const char* code = "") {
    return CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>>(derivedFn,
                                                                  code);
  }

  /// Specialization for differentiating functors.
  /// The specialization is needed because objects have to be passed
  /// by reference whereas functions have to be passed by value.
  template <unsigned... BitMaskedOpts, typename ArgSpec = const char*,
            typename F,
            typename DerivedFnType = ExtractDerivedFnTraitsForwMode_t<F>,
            typename = typename std::enable_if<
                !clad::HasOption(GetBitmaskedOpts(BitMaskedOpts...),
                                 opts::vector_mode) &&
                std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>> __attribute__((
      annotate("D")))
  differentiate(F&& f, ArgSpec args = "",
                DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
                const char* code = "") {
      return CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>>(derivedFn,
                                                                    code, f);
  }

  /// Generates function which computes derivative of `fn` argument w.r.t
  /// all parameters using a vectorized version of forward mode.
  ///
  /// \param[in] fn function to differentiate
  /// \param[in] args independent parameters information
  /// \returns `CladFunction` object to access the corresponding derived
  /// function.
  template <unsigned... BitMaskedOpts, typename ArgSpec = const char*,
            typename F,
            typename DerivedFnType = ExtractDerivedFnTraitsVecForwMode_t<F>,
            typename = typename std::enable_if<
                clad::HasOption(GetBitmaskedOpts(BitMaskedOpts...),
                                opts::vector_mode) &&
                !std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>, true> __attribute__((
      annotate("D")))
  differentiate(F fn, ArgSpec args = "",
                DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
                const char* code = "") {
    return CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>, true>(
        derivedFn, code);
  }

  /// Generates function which computes gradient of the given function wrt the
  /// parameters specified in `args` using reverse mode differentiation.
  ///
  /// \param[in] fn function to differentiate
  /// \param[in] args independent parameters information
  /// \returns `CladFunction` object to access the corresponding derived
  /// function.
  template <unsigned... BitMaskedOpts, typename ArgSpec = const char*,
            typename F, typename DerivedFnType = GradientDerivedFnTraits_t<F>,
            typename = typename std::enable_if<
                !std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>, true> __attribute__((
      annotate("G"))) CUDA_HOST_DEVICE
  gradient(F f, ArgSpec args = "",
           DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
           const char* code = "", bool CUDAkernel = false) {
    return CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>, true>(
        derivedFn /* will be replaced by gradient*/, code, nullptr, CUDAkernel);
  }

  /// Specialization for differentiating functors.
  /// The specialization is needed because objects have to be passed
  /// by reference whereas functions have to be passed by value.
  template <unsigned... BitMaskedOpts, typename ArgSpec = const char*,
            typename F, typename DerivedFnType = GradientDerivedFnTraits_t<F>,
            typename = typename std::enable_if<
                std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>, true> __attribute__((
      annotate("G"))) CUDA_HOST_DEVICE
  gradient(F&& f, ArgSpec args = "",
           DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
           const char* code = "") {
      return CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>, true>(
          derivedFn /* will be replaced by gradient*/, code, f);
  }

  /// Generates function which computes hessian matrix of the given function wrt
  /// the parameters specified in `args`.
  ///
  /// \param[in] fn function to differentiate
  /// \param[in] args independent parameters information
  /// \returns `CladFunction` object to access the corresponding derived
  /// function.
  template <unsigned... BitMaskedOpts, typename ArgSpec = const char*,
            typename F, typename DerivedFnType = HessianDerivedFnTraits_t<F>,
            typename = typename std::enable_if<
                !std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>> __attribute__((
      annotate("H")))
  hessian(F f, ArgSpec args = "",
          DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
          const char* code = "") {
    return CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>>(
        derivedFn /* will be replaced by hessian*/, code);
  }

  /// Specialization for differentiating functors.
  /// The specialization is needed because objects have to be passed
  /// by reference whereas functions have to be passed by value.
  template <unsigned... BitMaskedOpts, typename ArgSpec = const char*,
            typename F, typename DerivedFnType = HessianDerivedFnTraits_t<F>,
            typename = typename std::enable_if<
                std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>> __attribute__((
      annotate("H")))
  hessian(F&& f, ArgSpec args = "",
          DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
          const char* code = "") {
      return CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>>(
          derivedFn /* will be replaced by hessian*/, code, f);
  }

  /// Generates function which computes jacobian matrix of the given function
  /// wrt the parameters specified in `args` using reverse mode differentiation.
  ///
  /// \param[in] fn function to differentiate
  /// \param[in] args independent parameters information
  /// \returns `CladFunction` object to access the corresponding derived
  /// function.
  template <unsigned... BitMaskedOpts, typename ArgSpec = const char*,
            typename F, typename DerivedFnType = JacobianDerivedFnTraits_t<F>,
            typename = typename std::enable_if<
                !std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>> __attribute__((
      annotate("J")))
  jacobian(F f, ArgSpec args = "",
           DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
           const char* code = "") {
    return CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>>(
        derivedFn /* will be replaced by Jacobian*/, code);
  }

  /// Specialization for differentiating functors.
  /// The specialization is needed because objects have to be passed
  /// by reference whereas functions have to be passed by value.
  template <unsigned... BitMaskedOpts, typename ArgSpec = const char*,
            typename F, typename DerivedFnType = JacobianDerivedFnTraits_t<F>,
            typename = typename std::enable_if<
                std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>> __attribute__((
      annotate("J")))
  jacobian(F&& f, ArgSpec args = "",
           DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
           const char* code = "") {
      return CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>>(
          derivedFn /* will be replaced by Jacobian*/, code, f);
  }

  template <typename ArgSpec = const char*, typename F,
            typename DerivedFnType = GradientDerivedEstFnTraits_t<F>>
  CladFunction<DerivedFnType> __attribute__((annotate("E")))
  estimate_error(F f, ArgSpec args = "",
                 DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
                 const char* code = "") {
    return CladFunction<
        DerivedFnType>(derivedFn /* will be replaced by estimation code*/,
                       code);
  }

  // Gradient Structure for Reverse Mode Enzyme
  template <unsigned N> struct EnzymeGradient { double d_arr[N]; };
  } // namespace clad
#endif // CLAD_DIFFERENTIATOR

// Enable clad after the header was included.
// FIXME: The header inclusion should be made automatic if the pragma is seen.
#pragma clad ON
