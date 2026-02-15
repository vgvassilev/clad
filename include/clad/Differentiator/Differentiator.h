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
#ifdef __CUDACC__
#include "BuiltinDerivativesCUDA.cuh"
#endif
#include "CladConfig.h"
#include "FunctionTraits.h"
#include "Matrix.h"
#include "NumericalDiff.h"
#include "RestoreTracker.h"
#include "Tape.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <utility>
#ifndef __CUDACC__
#include <mutex>
#endif

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
template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024,
          bool is_multithread = false, bool DiskOffload = false>
using tape = tape_impl<T, SBO_SIZE, SLAB_SIZE, is_multithread, DiskOffload>;

/// Add value to the end of the tape, return the same value.
template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024,
          bool DiskOffload = false, typename... ArgsT>
CUDA_HOST_DEVICE T&
push(tape<T, SBO_SIZE, SLAB_SIZE, /*is_multithread=*/false, DiskOffload>& to,
     ArgsT... val) {
  to.emplace_back(std::forward<ArgsT>(val)...);
  return to.back();
}

/// A specialization for C arrays
template <typename T, typename U, size_t N, std::size_t SBO_SIZE = 64,
          std::size_t SLAB_SIZE = 1024, bool DiskOffload = false>
CUDA_HOST_DEVICE void
push(tape<T[N], SBO_SIZE, SLAB_SIZE, /*is_multithread=*/false, DiskOffload>& to,
     const U& val) {
  to.emplace_back();
  std::copy(std::begin(val), std::end(val), std::begin(to.back()));
}

  /// Remove the last value from the tape, return it.
template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024,
          bool DiskOffload = false>
CUDA_HOST_DEVICE T
pop(tape<T, SBO_SIZE, SLAB_SIZE, /*is_multithread=*/false, DiskOffload>& to) {
  T val = std::move(to.back());
  to.pop_back();
  return val;
}

  /// A specialization for C arrays
template <typename T, std::size_t N, std::size_t SBO_SIZE = 64,
          std::size_t SLAB_SIZE = 1024, bool DiskOffload = false>
CUDA_HOST_DEVICE void pop(tape<T[N], SBO_SIZE, SLAB_SIZE,
                               /*is_multithread=*/false, DiskOffload>& to) {
  to.pop_back();
}

  /// Access return the last value in the tape.
template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024,
          bool DiskOffload = false>
CUDA_HOST_DEVICE T&
back(tape<T, SBO_SIZE, SLAB_SIZE, /*is_multithread=*/false, DiskOffload>& of) {
  return of.back();
}

  /// Thread safe tape access functions with mutex locking mechanism
/// Thread safe tape access functions with mutex locking mechanism
#ifndef __CUDACC__
/// Add value to the end of the tape, return the same value.
template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024,
          bool DiskOffload = false, typename... ArgsT>
T push(tape<T, SBO_SIZE, SLAB_SIZE, /*is_multithreaded=*/true, DiskOffload>& to,
       ArgsT&&... val) {
  std::lock_guard<std::mutex> lock(to.mutex());
  to.emplace_back(std::forward<ArgsT>(val)...);
  return to.back();
}

  /// A specialization for C arrays
template <typename T, typename U, size_t N, std::size_t SBO_SIZE = 64,
          std::size_t SLAB_SIZE = 1024, bool DiskOffload = false>
void push(
    tape<T[N], SBO_SIZE, SLAB_SIZE, /*is_multithreaded=*/true, DiskOffload>& to,
    const U& val) {
  std::lock_guard<std::mutex> lock(to.mutex());
  to.emplace_back();
  std::copy(std::begin(val), std::end(val), std::begin(to.back()));
}

  /// Remove the last value from the tape, return it.
template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024,
          bool DiskOffload = false>
T pop(
    tape<T, SBO_SIZE, SLAB_SIZE, /*is_multithreaded=*/true, DiskOffload>& to) {
  std::lock_guard<std::mutex> lock(to.mutex());
  T val = std::move(to.back());
  to.pop_back();
  return val;
}

  /// A specialization for C arrays
template <typename T, std::size_t N, std::size_t SBO_SIZE = 64,
          std::size_t SLAB_SIZE = 1024, bool DiskOffload = false>
void pop(tape<T[N], SBO_SIZE, SLAB_SIZE, /*is_multithreaded=*/true,
              DiskOffload>& to) {
  std::lock_guard<std::mutex> lock(to.mutex());
  to.pop_back();
}

  /// Access return the last value in the tape.
template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024,
          bool DiskOffload = false>
T& back(
    tape<T, SBO_SIZE, SLAB_SIZE, /*is_multithreaded=*/true, DiskOffload>& of) {
  std::lock_guard<std::mutex> lock(of.mutex());
  return of.back();
}
#endif

  /// The purpose of this function is to initialize adjoints
  /// (or all of its iteratable elements) with 0.
  namespace zero_init_detail {
  template <class T> struct iterator_traits : std::iterator_traits<T> {};
  template <> struct iterator_traits<void*> {};
  template <> struct iterator_traits<const void*> {};

  template <class T, class It>
  std::integral_constant<
      bool, !std::is_same<typename std::remove_cv<T>::type,
                          typename iterator_traits<It>::value_type>::value>
  is_range_check(It first, It last);

  template <class T>
  decltype(is_range_check<T>(std::begin(std::declval<const T&>()),
                             std::end(std::declval<const T&>())))
  is_range(int);
  template <class T> std::false_type is_range(...);
  } // namespace zero_init_detail

  template <class T>
  struct is_range : decltype(zero_init_detail::is_range<T>(0)) {};

  template <class T> CUDA_HOST_DEVICE void zero_init(T& t);

  template <class T,
            typename std::enable_if<!is_range<T>::value, int>::type = 0>
  CUDA_HOST_DEVICE void zero_impl(volatile T& t) {
    // Fill an array with zeros.
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
    unsigned char tmp[sizeof(T)] = {};
    // Transfer the zeros with the magic function memcpy which can implicitly
    // create objects in the destination region of storage immediately prior to
    // copying the sequence of characters to the destination [27.5.1(3)].
    // (C++ has deprecated the volatile qualifiers. However, we drop them here
    // to make sure things still work with codebases which still have them)
    std::memcpy(const_cast<T*>(&t), tmp, sizeof(T));
  }

  template <class T, typename std::enable_if<is_range<T>::value, int>::type = 0>
  CUDA_HOST_DEVICE void zero_impl(T& t) {
    for (auto& x : t)
      zero_init(x);
  }

  template <class T> CUDA_HOST_DEVICE void zero_init(T& t) { zero_impl(t); }

  /// Initialize a const sized array.
  // NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays)
  template <typename T> CUDA_HOST_DEVICE void zero_init(T* x, std::size_t N) {
    for (std::size_t i = 0; i < N; ++i)
      zero_init(x[i]);
  }

  // This function is similar to the iterator-based std::move but is designed to
  // work with CUDA.
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  // An overload to initialize arrays from buffers, e.g., `clad::move(t0, arr)`
  template <class T, size_t N>
  CUDA_HOST_DEVICE void move(T* Input, T (&Output)[N]) {
    for (size_t i = 0; i < N; ++i)
      Output[i] = std::move(Input[i]);
  }

  // An overload to initialize arrays with init lists, e.g., `clad::move({1, 2},
  // arr)`
  template <class T, size_t N>
  CUDA_HOST_DEVICE void move(std::initializer_list<T> Input, T (&Output)[N]) {
    size_t i = 0;
    for (auto it = Input.begin(); it != Input.end() && i < N; ++it, ++i)
      Output[i] = *it;
    for (; i < N; ++i)
      Output[i] = T();
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
  CLAD_CONSTEXPR_CXX14 CUDA_HOST_DEVICE return_type_t<F>
  execute_with_default_args(list<Rest...>, F f, list<fArgTypes...>,
                            CUDA_ARGS CUDA_REST_ARGS Args&&... args) {
#if defined(__CUDACC__) && !defined(__CUDA_ARCH__)
    if (CUDAkernel) {
      constexpr size_t totalArgs = sizeof...(args) + sizeof...(Rest);
      std::array<void*, totalArgs> argPtrs = {(void*)(&args)...,
                                              static_cast<Rest>(nullptr)...};

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
  CLAD_CONSTEXPR_CXX14 return_type_t<F>
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
  constexpr CUDA_HOST_DEVICE auto
  execute_with_default_args(list<Rest...>, ReturnType C::*f, Obj&& obj,
                            list<fArgTypes...>,
                            Args&&... args) -> return_type_t<decltype(f)> {
    return (static_cast<Obj>(obj).*f)((fArgTypes)(args)...,
                                      static_cast<Rest>(nullptr)...);
  }

  template <bool EnablePadding, class... Rest, class ReturnType, class C,
            class Obj, class... Args, class... fArgTypes,
            typename std::enable_if<!EnablePadding, bool>::type = true>
  CLAD_CONSTEXPR_CXX14 auto
  execute_with_default_args(list<Rest...>, ReturnType C::*f, Obj&& obj,
                            list<fArgTypes...>,
                            Args&&... args) -> return_type_t<decltype(f)> {
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
            bool EnablePadding = false, bool ImmediateMode = false>
  class CladFunction {
  public:
    using CladFunctionType = F;
    using FunctorType = FunctorT;

  private:
    CladFunctionType m_Function;
    const char* m_Code;
    FunctorType *m_Functor = nullptr;
    bool m_CUDAkernel = false;

  public:
#ifdef __cpp_concepts
    CUDA_HOST_DEVICE CladFunction(CladFunctionType f, const char* code,
                                  FunctorType* functor = nullptr,
                                  bool CUDAkernel = false)
      requires(!ImmediateMode)
        : m_Function(f), m_Functor(functor), m_CUDAkernel(CUDAkernel) {
#ifndef __CLAD__
      static_assert(false, "clad doesn't appear to be loaded; make sure that "
                           "you pass clad.so to clang.");
#endif
      size_t length = GetLength(code);
      m_Code = (char*)malloc(length + 1);
      if (m_Code)
        memcpy((void*)m_Code, code, length + 1);
      else
        fprintf(stderr, "Error: Failed to allocate memory for m_Code\n");
    }

    constexpr CUDA_HOST_DEVICE CladFunction(CladFunctionType f,
                                            FunctorType* functor = nullptr,
                                            bool CUDAkernel = false)
      requires(ImmediateMode)
        : m_Function(f), m_Code("<constexpr functions don't have support for "
                                "printing the derivative yet>"),
          m_Functor(functor), m_CUDAkernel(CUDAkernel) {
#ifndef __CLAD__
      static_assert(false, "clad doesn't appear to be loaded; make sure that "
                           "you pass clad.so to clang.");
#endif
    }
#else
    CUDA_HOST_DEVICE CladFunction(CladFunctionType f, const char* code,
                                  FunctorType* functor = nullptr,
                                  bool CUDAkernel = false)
        : m_Function(f), m_Functor(functor), m_CUDAkernel(CUDAkernel) {
#ifndef __CLAD__
      static_assert(false, "clad doesn't appear to be loaded; make sure that "
                           "you pass clad.so to clang.");
#endif
      size_t length = GetLength(code);
      m_Code = (char*)malloc(length + 1);
      if (m_Code)
        memcpy((void*)m_Code, code, length + 1);
      else
#ifdef __CUDACC__
        printf("stderr: Error: Failed to allocate memory for m_Code\n");
#else
        fprintf(stderr, "Error: Failed to allocate memory for m_Code\n");
#endif
    }
#endif

    /// Constructor overload for initializing `m_Functor` when functor
    /// is passed by reference.
    CUDA_HOST_DEVICE CladFunction(CladFunctionType f, const char* code,
                                  FunctorType& functor)
        : CladFunction(f, code, &functor) {};

    constexpr CUDA_HOST_DEVICE CladFunction(CladFunctionType f,
                                            FunctorType& functor)
        : CladFunction(f, &functor) {};

    // Intentionally leak m_Code, otherwise we have to link against c++ runtime,
    // i.e -lstdc++.
    //~CladFunction() { /*free(m_Code);*/ }

    constexpr CladFunctionType getFunctionPtr() const { return m_Function; }

    template <typename... Args, class FnType = CladFunctionType>
    typename std::enable_if<!std::is_same<FnType, NoFunction*>::value,
                            return_type_t<F>>::type
        CLAD_CONSTEXPR_CXX14 CUDA_HOST_DEVICE
        execute(Args&&... args) const {
      if (!m_Function)
        return static_cast<return_type_t<F>>(return_type_t<F>());
      if (m_CUDAkernel) {
        printf("Use execute_kernel() for global CUDA kernels\n");
        return static_cast<return_type_t<F>>(return_type_t<F>());
      }
      // here static_cast is used to achieve perfect forwarding
#ifdef __CUDACC__
      return execute_helper(m_Function, m_CUDAkernel, dim3(0), dim3(0),
                            std::forward<Args>(args)...);
#else
      return execute_helper(m_Function, std::forward<Args>(args)...);
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
                            return_type_t<F>>::type constexpr CUDA_HOST_DEVICE
    execute(Args&&... args) const {
      return static_cast<return_type_t<F>>(0);
    }

    template <typename... Args>
    constexpr CUDA_HOST_DEVICE auto operator()(Args&&... args) const
        -> decltype(this->execute(std::forward<Args>(args)...)) {
      return execute(std::forward<Args>(args)...);
    }

    /// Return the string representation for the generated derivative.
    CLAD_CONSTEXPR_CXX14 const char* getCode() const {
      if (m_Code)
        return m_Code;
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
      CLAD_CONSTEXPR_CXX14 CUDA_HOST_DEVICE return_type_t<CladFunctionType>
      execute_helper(Fn f, CUDA_ARGS Args&&... args) const {
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
      template <class ReturnType, class C, class Obj,
                class = typename std::enable_if<std::is_same<
                    typename std::decay<Obj>::type, C>::value>::type,
                class... Args>
      CLAD_CONSTEXPR_CXX14 return_type_t<CladFunctionType>
      execute_helper(ReturnType C::*f, Obj&& obj, Args&&... args) const {
        // `static_cast` is required here for perfect forwarding.
        return execute_with_default_args<EnablePadding>(
            DropArgs_t<sizeof...(Args), decltype(f)>{}, f,
            static_cast<Obj>(obj),
            TakeNFirstArgs_t<sizeof...(Args), decltype(f)>{},
            static_cast<Args>(args)...);
      }
      /// If user have not passed object explicitly, then this specialization
      /// will be used and derived function will be called through the object
      /// saved in `CladFunction`.
      template <class ReturnType, class C, class... Args>
      CLAD_CONSTEXPR_CXX14 return_type_t<CladFunctionType>
      execute_helper(ReturnType C::*f, Args&&... args) const {
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
                !clad::HasOption(GetBitmaskedOpts(BitMaskedOpts...),
                                 opts::immediate_mode) &&
                !std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>> __attribute__((
      annotate("D")))
  differentiate(F fn, ArgSpec args = "",
                DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
                const char* code = "") {
    return CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>>(derivedFn,
                                                                  code);
  }

  template <unsigned... BitMaskedOpts, typename ArgSpec = const char*,
            typename F,
            typename DerivedFnType = ExtractDerivedFnTraitsForwMode_t<F>,
            typename = typename std::enable_if<
                !clad::HasOption(GetBitmaskedOpts(BitMaskedOpts...),
                                 opts::vector_mode) &&
                clad::HasOption(GetBitmaskedOpts(BitMaskedOpts...),
                                opts::immediate_mode) &&
                !std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  constexpr CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>, false,
                         true> __attribute__((annotate("D")))
  differentiate(F fn, ArgSpec args = "",
                DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr)) {
    return CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>, false, true>(
        derivedFn);
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
  constexpr CladFunction<
      DerivedFnType, ExtractFunctorTraits_t<F>> __attribute__((annotate("D")))
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
  constexpr CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>,
                         true> __attribute__((annotate("D")))
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
                !clad::HasOption(GetBitmaskedOpts(BitMaskedOpts...),
                                 opts::immediate_mode) &&
                !std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  constexpr CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>,
                         true> __attribute__((annotate("G"))) CUDA_HOST_DEVICE
  gradient(F f, ArgSpec args = "",
           DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
           const char* code = "", bool CUDAkernel = false) {
    return CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>, true>(
        derivedFn /* will be replaced by gradient*/, code, nullptr, CUDAkernel);
  }

  template <unsigned... BitMaskedOpts, typename ArgSpec = const char*,
            typename F, typename DerivedFnType = GradientDerivedFnTraits_t<F>,
            typename = typename std::enable_if<
                clad::HasOption(GetBitmaskedOpts(BitMaskedOpts...),
                                opts::immediate_mode) &&
                !std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  constexpr CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>, true,
                         true> __attribute__((annotate("G"))) CUDA_HOST_DEVICE
  gradient(F f, ArgSpec args = "",
           DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
           bool CUDAkernel = false) {
    return CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>, true, true>(
        derivedFn /* will be replaced by gradient*/, nullptr, CUDAkernel);
  }

  /// Specialization for differentiating functors.
  /// The specialization is needed because objects have to be passed
  /// by reference whereas functions have to be passed by value.
  template <unsigned... BitMaskedOpts, typename ArgSpec = const char*,
            typename F, typename DerivedFnType = GradientDerivedFnTraits_t<F>,
            typename = typename std::enable_if<
                std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  constexpr CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>,
                         true> __attribute__((annotate("G"))) CUDA_HOST_DEVICE
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
                !clad::HasOption(GetBitmaskedOpts(BitMaskedOpts...),
                                 opts::immediate_mode) &&
                !std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  constexpr CladFunction<
      DerivedFnType, ExtractFunctorTraits_t<F>> __attribute__((annotate("H")))
  hessian(F f, ArgSpec args = "",
          DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
          const char* code = "") {
    return CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>>(
        derivedFn /* will be replaced by hessian*/, code);
  }

  template <unsigned... BitMaskedOpts, typename ArgSpec = const char*,
            typename F, typename DerivedFnType = HessianDerivedFnTraits_t<F>,
            typename = typename std::enable_if<
                clad::HasOption(GetBitmaskedOpts(BitMaskedOpts...),
                                opts::immediate_mode) &&
                !std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  constexpr CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>, false,
                         true> __attribute__((annotate("H")))
  hessian(F f, ArgSpec args = "",
          DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr)) {
    return CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>, false, true>(
        derivedFn /* will be replaced by hessian*/);
  }

  /// Specialization for differentiating functors.
  /// The specialization is needed because objects have to be passed
  /// by reference whereas functions have to be passed by value.
  template <unsigned... BitMaskedOpts, typename ArgSpec = const char*,
            typename F, typename DerivedFnType = HessianDerivedFnTraits_t<F>,
            typename = typename std::enable_if<
                std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  constexpr CladFunction<
      DerivedFnType, ExtractFunctorTraits_t<F>> __attribute__((annotate("H")))
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
  constexpr CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>,
                         /*EnablePadding=*/true> __attribute__((annotate("J")))
  jacobian(F f, ArgSpec args = "",
           DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
           const char* code = "") {
    return CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>,
                        /*EnablePadding=*/true>(
        derivedFn /* will be replaced by Jacobian*/, code);
  }

  /// Specialization for differentiating functors.
  /// The specialization is needed because objects have to be passed
  /// by reference whereas functions have to be passed by value.
  template <unsigned... BitMaskedOpts, typename ArgSpec = const char*,
            typename F, typename DerivedFnType = JacobianDerivedFnTraits_t<F>,
            typename = typename std::enable_if<
                std::is_class<remove_reference_and_pointer_t<F>>::value>::type>
  constexpr CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>,
                         /*EnablePadding=*/true> __attribute__((annotate("J")))
  jacobian(F&& f, ArgSpec args = "",
           DerivedFnType derivedFn = static_cast<DerivedFnType>(nullptr),
           const char* code = "") {
    return CladFunction<DerivedFnType, ExtractFunctorTraits_t<F>,
                        /*EnablePadding=*/true>(
        derivedFn /* will be replaced by Jacobian*/, code, f);
  }

  template <typename ArgSpec = const char*, typename F,
            typename DerivedFnType = GradientDerivedEstFnTraits_t<F>>
  constexpr CladFunction<DerivedFnType> __attribute__((annotate("E")))
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
