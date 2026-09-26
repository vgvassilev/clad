#ifndef FUNCTION_TRAITS
#define FUNCTION_TRAITS

#include "clad/Differentiator/ArrayRef.h"
#include "clad/Differentiator/Matrix.h"

#include <type_traits>

namespace clad {
  /// Utility type trait to remove both reference and pointer
  /// from type `T`.
  ///
  /// First removes any reference qualifier associated with `T`,
  /// then removes any associated pointer. Resulting type is provided
  /// as member typedef `type`.
  template <typename T> struct remove_reference_and_pointer {
    using type = typename std::remove_pointer<
        typename std::remove_reference<T>::type>::type;
  };

  /// Helper type for remove_reference_and_pointer.
  template <typename T>
  using remove_reference_and_pointer_t =
      typename remove_reference_and_pointer<T>::type;

  /// Check whether class `C` defines a call operator. Provides the member
  /// constant `value` which is equal to true, if class defines call operator.
  /// Otherwise `value` is equal to false.
  template <typename C, typename = void>
  struct has_call_operator : std::false_type {};

  /// \cond DOXYGEN_CANNOT_PARSE_THIS
  template <typename C>
  struct has_call_operator<
      C,
      typename std::enable_if<(
          sizeof(&remove_reference_and_pointer_t<C>::operator()) > 0)>::type>
      : std::true_type {};
  /// \endcond

  /// Placeholder type for denoting no function type exists
  ///
  /// This is used by `ExtractDerivedFnTraitsForwMode` and 
  /// `ExtractDerivedFnTraits` type trait as value for member typedef
  /// `type` to denote no function type exists.
  class NoFunction {};

  template <typename... T> struct list {};

  // Trait class to deduce return type of function(both member and non-member) at commpile time
  // Only function pointer types are supported by this trait class
  template <class F> struct function_traits {};
  template <class F>
  using return_type_t = typename function_traits<F>::return_type;
  template <class F>
  using argument_types_t = typename function_traits<F>::argument_types;

  // specializations for non-member functions pointer types
  template <class ReturnType, class... Args>
  struct function_traits<ReturnType (*)(Args...)> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class... Args>
  struct function_traits<ReturnType (*)(Args..., ...)> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };

  // specializations for member functions pointer types with no qualifiers
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...)> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...)> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };

  // specializations for member functions pointer type with only cv-qualifiers
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) const> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) const> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) volatile> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) volatile> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) const volatile> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) const volatile> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };

  // specializations for member functions pointer types with 
  // reference qualifiers and with and without cv-qualifiers
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...)&> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...)&> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) const&> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) const&> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) volatile&> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) volatile&> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) const volatile&> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) const volatile&> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) &&> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) &&> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) const&&> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) const&&> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) volatile&&> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) volatile&&> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) const volatile&&> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) const volatile&&> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };

  template <> struct function_traits<NoFunction*> {
    using return_type = void;
    using argument_types = void;
  };

  // specializations for noexcept member functions
  #if __cpp_noexcept_function_type > 0
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) const noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) const noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) volatile noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) volatile noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) const volatile noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...)
                             const volatile noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) & noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) & noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) const & noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) const & noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) volatile & noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) volatile & noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) const volatile & noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) const volatile &
                         noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) && noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) && noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) const && noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) const && noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) volatile && noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) volatile && noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args...) const volatile &&
                         noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
  template <class ReturnType, class C, class... Args>
  struct function_traits<ReturnType (C::*)(Args..., ...) const volatile &&
                         noexcept> {
    using return_type = ReturnType;
    using argument_types = list<Args...>;
  };
#endif

#define REM_CTOR(...) __VA_ARGS__

  // Setup for DropArgs
  struct dummy {};

  template <typename T> struct wrap {
    constexpr operator dummy() const { return {}; }
  };

  template <std::size_t I> using placeholder = dummy;

  template <size_t... Ints> struct IndexSequence {
    using type = IndexSequence;
    using value_type = size_t;
    static constexpr std::size_t size() noexcept { return sizeof...(Ints); }
  };

  template <class Sequence1, class Sequence2> struct MergeAndRemember;

  template <size_t... I1, size_t... I2>
  struct MergeAndRemember<IndexSequence<I1...>, IndexSequence<I2...>>
      : IndexSequence<I1..., (sizeof...(I1) + I2)...> {};

  // Creates a sequence of numbers in the template in place of
  template <size_t N>
  struct MakeIndexSequence
      : MergeAndRemember<typename MakeIndexSequence<N / 2>::type,
                         typename MakeIndexSequence<N - N / 2>::type> {};

  template <> struct MakeIndexSequence<0> : IndexSequence<> {};
  template <> struct MakeIndexSequence<1> : IndexSequence<0> {};

  template <std::size_t... Idx, typename... Rest>
  auto wrapRest(placeholder<Idx>..., wrap<Rest>...) -> list<Rest...>;

  template <typename... T, std::size_t... Idx>
  auto dropUsingIndex(IndexSequence<Idx...>)
      -> decltype(wrapRest<Idx...>(wrap<T>{}...)) {
    return wrapRest<Idx...>(wrap<T>{}...);
  }

  template <std::size_t N, typename... T>
  using Drop_t = decltype(dropUsingIndex<T...>(MakeIndexSequence<N>{}));

  template <std::size_t N, typename... Args>
  constexpr auto DropArgs(list<Args...>) -> Drop_t<N, Args...>;

  // Returns the Args in the function F that come after the Nth arg
  template <std::size_t N, typename F>
  using DropArgs_t = decltype(DropArgs<N>(argument_types_t<F>{}));

  template <typename... Args, std::size_t... Idx>
  constexpr auto TakeNFirstArgs(IndexSequence<Idx...>)
      -> list<typename std::tuple_element<Idx, std::tuple<Args...>>::type...>;

  template <std::size_t N, typename... Args>
  constexpr auto TakeNFirstArgs(list<Args...>)
      -> decltype(TakeNFirstArgs<Args...>(MakeIndexSequence<N>{}));

  // Returns the first N arguments in the function F
  template <size_t N, typename F>
  using TakeNFirstArgs_t = decltype(TakeNFirstArgs<N>(argument_types_t<F>{}));

  template <class T, class R> struct OutputParamType {
    using type = typename std::remove_pointer<R>::type*;
  };

  template <class T, class R>
  using OutputParamType_t = typename OutputParamType<T, R>::type;

  template <class T, class = void> struct GradientDerivedFnTraits {};

  // GradientDerivedFnTraits is used to deduce type of the derived functions
  // derived using reverse modes
  template <class T>
  using GradientDerivedFnTraits_t = typename GradientDerivedFnTraits<T>::type;

  // GradientDerivedFnTraits specializations for pure function pointer types
  template <class ReturnType, class... Args>
  struct GradientDerivedFnTraits<ReturnType (*)(Args...)> {
    using type = void (*)(Args..., OutputParamType_t<Args, void>...);
  };

  /// These macro expansions are used to cover all possible cases of
  /// qualifiers in member functions when declaring GradientDerivedFnTraits.
  /// They need to be read from bottom to top. Starting from the use of AddCON,
  /// the call to which is used to pass the cases with and without C-style
  /// varargs, then as the macro name AddCON says it adds cases of const
  /// qualifier. The AddVOL and AddREF macro similarly add cases for volatile
  /// qualifier and reference respectively. The AddNOEX adds cases for noexcept
  /// qualifier only if it is supported and finally AddSPECS declares the
  /// function with all the cases
#define GradientDerivedFnTraits_AddSPECS(var, cv, vol, ref, noex)              \
  template <typename R, typename C, typename... Args>                          \
  struct GradientDerivedFnTraits<R (C::*)(Args...) cv vol ref noex> {          \
    using type =                                                               \
        void (C::*)(Args..., OutputParamType_t<C, void>,                       \
                    OutputParamType_t<Args, void>...) cv vol ref noex;         \
  };

#if __cpp_noexcept_function_type > 0
#define GradientDerivedFnTraits_AddNOEX(var, con, vol, ref)                    \
  GradientDerivedFnTraits_AddSPECS(var, con, vol, ref, )                       \
      GradientDerivedFnTraits_AddSPECS(var, con, vol, ref, noexcept)
#else
#define GradientDerivedFnTraits_AddNOEX(var, con, vol, ref)                    \
  GradientDerivedFnTraits_AddSPECS(var, con, vol, ref, )
#endif

#define GradientDerivedFnTraits_AddREF(var, con, vol)                          \
  GradientDerivedFnTraits_AddNOEX(var, con, vol, )                             \
      GradientDerivedFnTraits_AddNOEX(var, con, vol, &)                        \
          GradientDerivedFnTraits_AddNOEX(var, con, vol, &&)

#define GradientDerivedFnTraits_AddVOL(var, con)                               \
  GradientDerivedFnTraits_AddREF(var, con, )                                   \
      GradientDerivedFnTraits_AddREF(var, con, volatile)

#define GradientDerivedFnTraits_AddCON(var)                                    \
  GradientDerivedFnTraits_AddVOL(var, )                                        \
      GradientDerivedFnTraits_AddVOL(var, const)

  GradientDerivedFnTraits_AddCON(()); // Declares all the specializations

  /// Specialization for class types
  /// If class have exactly one user defined call operator, then defines
  /// member typedef `type` same as the type of the derived function of the
  /// call operator, otherwise defines member typedef `type` as the type of
  /// `NoFunction*`.
  template <class F>
  struct GradientDerivedFnTraits<
      F, typename std::enable_if<
             std::is_class<remove_reference_and_pointer_t<F>>::value &&
             has_call_operator<F>::value>::type> {
    using ClassType =
        typename std::decay<remove_reference_and_pointer_t<F>>::type;
    using type = GradientDerivedFnTraits_t<decltype(&ClassType::operator())>;
  };
  template <class F>
  struct GradientDerivedFnTraits<
      F, typename std::enable_if<
             std::is_class<remove_reference_and_pointer_t<F>>::value &&
             !has_call_operator<F>::value>::type> {
    using type = NoFunction*;
  };

  /// This specific specialization is for error estimation calls.
  template <class T, class = void> struct GradientDerivedEstFnTraits {};

  // GradientDerivedEstFnTraits is used to deduce type of the derived functions
  // derived using reverse modes
  template <class T>
  using GradientDerivedEstFnTraits_t = typename GradientDerivedEstFnTraits<
      T>::type;

  // GradientDerivedEstFnTraits specializations for pure function pointer types
  template <class ReturnType, class... Args>
  struct GradientDerivedEstFnTraits<ReturnType (*)(Args...)> {
    using type = void (*)(Args..., OutputParamType_t<Args, Args>...,
                          double&);
  };

  /// These macro expansions are used to cover all possible cases of
  /// qualifiers in member functions when declaring GradientDerivedFnTraits.
  /// They need to be read from bottom to top. Starting from the use of AddCON,
  /// the call to which is used to pass the cases with and without C-style
  /// varargs, then as the macro name AddCON says it adds cases of const
  /// qualifier. The AddVOL and AddREF macro similarly add cases for volatile
  /// qualifier and reference respectively. The AddNOEX adds cases for noexcept
  /// qualifier only if it is supported and finally AddSPECS declares the
  /// function with all the cases
#define GradientDerivedEstFnTraits_AddSPECS(var, cv, vol, ref, noex)           \
  template <typename R, typename C, typename... Args>                          \
  struct GradientDerivedEstFnTraits<R (C::*)(Args...) cv vol ref noex> {       \
    using type = void (C::*)(Args..., OutputParamType_t<Args, Args>...,           \
                             double&) cv vol ref noex;                         \
  };

#if __cpp_noexcept_function_type > 0
#define GradientDerivedEstFnTraits_AddNOEX(var, con, vol, ref)                 \
  GradientDerivedEstFnTraits_AddSPECS(var, con, vol, ref, )                    \
      GradientDerivedEstFnTraits_AddSPECS(var, con, vol, ref, noexcept)
#else
#define GradientDerivedEstFnTraits_AddNOEX(var, con, vol, ref)                 \
  GradientDerivedEstFnTraits_AddSPECS(var, con, vol, ref, )
#endif

#define GradientDerivedEstFnTraits_AddREF(var, con, vol)                       \
  GradientDerivedEstFnTraits_AddNOEX(var, con, vol, )                          \
      GradientDerivedEstFnTraits_AddNOEX(var, con, vol, &)                     \
          GradientDerivedEstFnTraits_AddNOEX(var, con, vol, &&)

#define GradientDerivedEstFnTraits_AddVOL(var, con)                            \
  GradientDerivedEstFnTraits_AddREF(var, con, )                                \
      GradientDerivedEstFnTraits_AddREF(var, con, volatile)

#define GradientDerivedEstFnTraits_AddCON(var)                                 \
  GradientDerivedEstFnTraits_AddVOL(var, )                                     \
      GradientDerivedEstFnTraits_AddVOL(var, const)

  GradientDerivedEstFnTraits_AddCON(()); // Declares all the specializations

  /// Specialization for class types
  /// If class have exactly one user defined call operator, then defines
  /// member typedef `type` same as the type of the derived function of the
  /// call operator, otherwise defines member typedef `type` as the type of
  /// `NoFunction*`.
  template <class F>
  struct GradientDerivedEstFnTraits<
      F, typename std::enable_if<
             std::is_class<remove_reference_and_pointer_t<F>>::value &&
             has_call_operator<F>::value>::type> {
    using ClassType = typename std::decay<
        remove_reference_and_pointer_t<F>>::type;
    using type = GradientDerivedEstFnTraits_t<decltype(&ClassType::operator())>;
  };
  template <class F>
  struct GradientDerivedEstFnTraits<
      F, typename std::enable_if<
             std::is_class<remove_reference_and_pointer_t<F>>::value &&
             !has_call_operator<F>::value>::type> {
    using type = NoFunction*;
  };

  // OutputVecParamType is used to deduce the type of derivative arguments
  // for vector forward mode.
  template <class T, class R> struct OutputVecParamType {
    using type = array_ref<typename std::remove_pointer<R>::type>;
  };

  template <class T, class R>
  using OutputVecParamType_t = typename OutputVecParamType<T, R>::type;

  /// Specialization for vector forward mode type.
  template <class F, class = void> struct ExtractDerivedFnTraitsVecForwMode {};

  template <class F>
  using ExtractDerivedFnTraitsVecForwMode_t =
      typename ExtractDerivedFnTraitsVecForwMode<F>::type;

  template <class ReturnType, class... Args>
  struct ExtractDerivedFnTraitsVecForwMode<ReturnType (*)(Args...)> {
    using type = void (*)(Args..., OutputVecParamType_t<Args, void>...);
  };

  template <class T, class = void> struct JacobianDerivedFnTraits {};

  // JacobianDerivedFnTraits is used to deduce type of the derived functions
  // derived using jacobian mode
  template <class T>
  using JacobianDerivedFnTraits_t = typename JacobianDerivedFnTraits<T>::type;

  // JacobianDerivedFnTraits specializations for pure function pointer types
  template <class ReturnType, class... Args>
  struct JacobianDerivedFnTraits<ReturnType (*)(Args...)> {
    using type = void (*)(Args..., OutputParamType_t<Args, void>...);
  };

  /// These macro expansions are used to cover all possible cases of
  /// qualifiers in member functions when declaring JacobianDerivedFnTraits.
  /// They need to be read from bottom to top. Starting from the use of AddCON,
  /// the call to which is used to pass the cases with and without C-style
  /// varargs, then as the macro name AddCON says it adds cases of const
  /// qualifier. The AddVOL and AddREF macro similarly add cases for volatile
  /// qualifier and reference respectively. The AddNOEX adds cases for noexcept
  /// qualifier only if it is supported and finally AddSPECS declares the
  /// function with all the cases
#define JacobianDerivedFnTraits_AddSPECS(var, cv, vol, ref, noex)            \
    template <typename R, typename C, typename... Args>                        \
    struct JacobianDerivedFnTraits<R (C::*)(Args...) cv vol ref noex> {        \
      using type = void (C::*)(                                                \
          Args..., OutputParamType_t<Args, void>...) cv vol ref noex;          \
    };

#if __cpp_noexcept_function_type > 0
#define JacobianDerivedFnTraits_AddNOEX(var, con, vol, ref)                    \
  JacobianDerivedFnTraits_AddSPECS(var, con, vol, ref, )                       \
      JacobianDerivedFnTraits_AddSPECS(var, con, vol, ref, noexcept)
#else
#define JacobianDerivedFnTraits_AddNOEX(var, con, vol, ref)                    \
  JacobianDerivedFnTraits_AddSPECS(var, con, vol, ref, )
#endif

#define JacobianDerivedFnTraits_AddREF(var, con, vol)                          \
  JacobianDerivedFnTraits_AddNOEX(var, con, vol, )                             \
      JacobianDerivedFnTraits_AddNOEX(var, con, vol, &)                        \
          JacobianDerivedFnTraits_AddNOEX(var, con, vol, &&)

#define JacobianDerivedFnTraits_AddVOL(var, con)                               \
  JacobianDerivedFnTraits_AddREF(var, con, )                                   \
      JacobianDerivedFnTraits_AddREF(var, con, volatile)

#define JacobianDerivedFnTraits_AddCON(var)                                    \
  JacobianDerivedFnTraits_AddVOL(var, )                                        \
      JacobianDerivedFnTraits_AddVOL(var, const)

  JacobianDerivedFnTraits_AddCON(()); // Declares all the specializations

  /// Specialization for class types
  /// If class have exactly one user defined call operator, then defines
  /// member typedef `type` same as the type of the derived function of the
  /// call operator, otherwise defines member typedef `type` as the type of
  /// `NoFunction*`.
  template <class F>
  struct JacobianDerivedFnTraits<
      F, typename std::enable_if<
             std::is_class<remove_reference_and_pointer_t<F>>::value &&
             has_call_operator<F>::value>::type> {
    using ClassType = typename std::decay<
        remove_reference_and_pointer_t<F>>::type;
    using type = JacobianDerivedFnTraits_t<decltype(&ClassType::operator())>;
  };
  template <class F>
  struct JacobianDerivedFnTraits<
      F, typename std::enable_if<
             std::is_class<remove_reference_and_pointer_t<F>>::value &&
             !has_call_operator<F>::value>::type> {
    using type = NoFunction*;
  };

  template <class T, class = void> struct HessianDerivedFnTraits {};

  // HessianDerivedFnTraits is used to deduce type of the derived functions
  // derived using hessian mode
  template <class T>
  using HessianDerivedFnTraits_t = typename HessianDerivedFnTraits<T>::type;

  // HessianDerivedFnTraits specializations for pure function pointer types
  template <class ReturnType, class... Args>
  struct HessianDerivedFnTraits<ReturnType (*)(Args...)> {
    using type = void (*)(Args..., ReturnType*);
  };

  /// These macro expansions are used to cover all possible cases of
  /// qualifiers in member functions when declaring HessianDerivedFnTraits.
  /// They need to be read from bottom to top. Starting from the use of AddCON,
  /// the call to which is used to pass the cases with and without C-style
  /// varargs, then as the macro name AddCON says it adds cases of const
  /// qualifier. The AddVOL and AddREF macro similarly add cases for volatile
  /// qualifier and reference respectively. The AddNOEX adds cases for noexcept
  /// qualifier only if it is supported and finally AddSPECS declares the
  /// function with all the cases
#define HessianDerivedFnTraits_AddSPECS(var, cv, vol, ref, noex)               \
  template <typename R, typename C, typename... Args>                          \
  struct HessianDerivedFnTraits<R (C::*)(Args...) cv vol ref noex> {           \
    using type = void (C::*)(Args..., R*) cv vol ref noex;                     \
  };

#if __cpp_noexcept_function_type > 0
#define HessianDerivedFnTraits_AddNOEX(var, con, vol, ref)                     \
  HessianDerivedFnTraits_AddSPECS(var, con, vol, ref, )                        \
      HessianDerivedFnTraits_AddSPECS(var, con, vol, ref, noexcept)
#else
#define HessianDerivedFnTraits_AddNOEX(var, con, vol, ref)                     \
  HessianDerivedFnTraits_AddSPECS(var, con, vol, ref, )
#endif

#define HessianDerivedFnTraits_AddREF(var, con, vol)                           \
  HessianDerivedFnTraits_AddNOEX(var, con, vol, )                              \
      HessianDerivedFnTraits_AddNOEX(var, con, vol, &)                         \
          HessianDerivedFnTraits_AddNOEX(var, con, vol, &&)

#define HessianDerivedFnTraits_AddVOL(var, con)                                \
  HessianDerivedFnTraits_AddREF(var, con, )                                    \
      HessianDerivedFnTraits_AddREF(var, con, volatile)

#define HessianDerivedFnTraits_AddCON(var)                                     \
  HessianDerivedFnTraits_AddVOL(var, ) HessianDerivedFnTraits_AddVOL(var, const)

  HessianDerivedFnTraits_AddCON(()); // Declares all the specializations

  /// Specialization for class types
  /// If class have exactly one user defined call operator, then defines
  /// member typedef `type` same as the type of the derived function of the
  /// call operator, otherwise defines member typedef `type` as the type of
  /// `NoFunction*`.
  template <class F>
  struct HessianDerivedFnTraits<
      F, typename std::enable_if<
             std::is_class<remove_reference_and_pointer_t<F>>::value &&
             has_call_operator<F>::value>::type> {
    using ClassType = typename std::decay<
        remove_reference_and_pointer_t<F>>::type;
    using type = HessianDerivedFnTraits_t<decltype(&ClassType::operator())>;
  };
  template <class F>
  struct HessianDerivedFnTraits<
      F, typename std::enable_if<
             std::is_class<remove_reference_and_pointer_t<F>>::value &&
             !has_call_operator<F>::value>::type> {
    using type = NoFunction*;
  };

  /// Compute type of derived function of function, method or functor when
  /// differentiated using forward differentiation mode
  /// (`clad::differentiate`). Computed type is provided as member typedef
  /// `type`.
  ///
  /// More precisely, this type trait behaves as following:
  ///
  /// - If `F` is a function pointer type
  ///   Defines member typedef `type` same as the type of the function
  ///   pointer.
  ///
  /// - If `F` is a member function pointer type
  ///   Defines member typedef `type` same as the type of the member
  ///   function
  /// pointer.
  ///
  /// - If `F` is class type, class reference type, class pointer type, or
  ///   reference to class pointer type.
  ///   Defines member typedef `type` same as the type of the overloaded
  ///   call operator member function of the class.
  ///
  /// - For all other cases, no member typedef `type` is provided.
  ///
  /// This type trait is specific to forward mode differentiation since the
  /// rules for computing the signature of derived functions are different
  /// for forward and reverse mode.
  template <class F, class = void> struct ExtractDerivedFnTraitsForwMode {};

  /// Helper type for ExtractDerivedFnTraitsForwMode
  template <class F>
  using ExtractDerivedFnTraitsForwMode_t =
      typename ExtractDerivedFnTraitsForwMode<F>::type;

  /// Whether a parameter is one a function writes a result through: a
  /// non-const lvalue reference to a floating point type. Kept in step with
  /// utils::CollectOutputParams, which decides the same thing on the AST side
  /// and has to reach the same answer -- where the two disagree, one side
  /// gives the derivative a return type the other does not know about.
  /// Floating point rather than arithmetic, so that an `int&` a function
  /// counts iterations through is not mistaken for a second output.
  ///
  /// The `std::..._v` spelling is not available to these traits: part of the
  /// test suite compiles this header as C++14, where the variable templates do
  /// not exist. The `_t` aliases are C++14 and are used. This is what the
  /// NOLINTs below are for -- modernize-type-traits asks for `_v` here, and
  /// taking the suggestion breaks the C++14 rows.
  template <class T> struct IsOutputParam : std::false_type {};
  // NOLINTBEGIN(modernize-type-traits)
  template <class T>
  struct IsOutputParam<T&>
      : std::integral_constant<bool, std::is_floating_point<T>::value &&
                                         !std::is_const<T>::value> {};
  // NOLINTEND(modernize-type-traits)

  /// The tangent type of the sole output parameter among `Ts`, or void when
  /// there is not exactly one of them. `Found` carries the match made so far,
  /// void meaning none yet; a second match collapses the answer back to void,
  /// since no one choice among several outputs is the right one. Written
  /// recursively rather than as a fold expression because these headers are
  /// compiled as C++14 by part of the test suite.
  template <class Found, class... Ts> struct SoleOutputTangent {
    using type = Found;
  };
  // NOLINTBEGIN(modernize-type-traits)
  template <class Found, class T, class... Rest>
  struct SoleOutputTangent<Found, T, Rest...> {
    using OnMatch = std::conditional_t<
        std::is_same<Found, void>::value,
        SoleOutputTangent<std::remove_const_t<std::remove_reference_t<T>>,
                          Rest...>,
        SoleOutputTangent<void>>;
    using type =
        typename std::conditional_t<IsOutputParam<T>::value, OnMatch,
                                    SoleOutputTangent<Found, Rest...>>::type;
  };
  // NOLINTEND(modernize-type-traits)

  /// The signature a forward-mode derivative is given. It is the primal's,
  /// except that a primal returning void returns the tangent of its sole
  /// output parameter instead -- otherwise that tangent is computed into a
  /// local the caller cannot reach. Mirrors utils::GetDerivativeType.
  template <class F> struct ForwardModeDerivedSignature {
    using type = F;
  };
  template <class... Args> struct ForwardModeDerivedSignature<void(Args...)> {
    using type = typename SoleOutputTangent<void, Args...>::type(Args...);
  };
  /// A C-variadic primal is the same question with an ellipsis to carry over.
  /// Without this specialization such a function falls back to the primal's
  /// own type while the generated code returns a tangent, and the two
  /// disagreeing is what execute reads a stale register through.
  template <class... Args>
  struct ForwardModeDerivedSignature<void(Args..., ...)> {
    using type = typename SoleOutputTangent<void, Args...>::type(Args..., ...);
  };

  /// Specialization for free function pointer type
  template <class F>
  struct ExtractDerivedFnTraitsForwMode<
      F*, typename std::enable_if<std::is_function<F>::value>::type> {
    using type = typename ForwardModeDerivedSignature<
        remove_reference_and_pointer_t<F>>::type*;
  };

  /// Specialization for member function pointer type
  template <class F>
  struct ExtractDerivedFnTraitsForwMode<
      F, typename std::enable_if<
             std::is_member_function_pointer<F>::value>::type> {
    using type = typename std::decay<F>::type;
  };

  /// Specialization for class types
  /// If class have exactly one user defined call operator, then defines
  /// member typedef `type` same as the type of the call operator, otherwise
  /// defines member typedef `type` as the type of `NoFunction*`.
  template <class F>
  struct ExtractDerivedFnTraitsForwMode<
      F, typename std::enable_if<
             std::is_class<remove_reference_and_pointer_t<F>>::value &&
             has_call_operator<F>::value>::type> {
    using ClassType =
        typename std::decay<remove_reference_and_pointer_t<F>>::type;
    using type = decltype(&ClassType::operator());
  };

  template <class F>
  struct ExtractDerivedFnTraitsForwMode<
      F, typename std::enable_if<
             std::is_class<remove_reference_and_pointer_t<F>>::value &&
             !has_call_operator<F>::value>::type> {
    using type = NoFunction*;
  };

  /// Placeholder type for denoting no object type exists.
  ///
  /// This is used by `ExtractFunctorTraits` type trait as value of member
  /// typedef `type` to denote no functor type exist.
  class NoObject {};

  /// Compute class type from member function type, deduced type is
  /// void if free function type is provided. If class type is provided,
  /// then deduced type is same as that of the provided class type.
  ///
  /// More precisely, this type trait behaves as following :
  ///
  /// - If `F` is a function pointer type
  ///   Defines member typedef `type` as the type of class `NoObject`.
  ///
  /// - If `F` is a memeber function pointer type
  ///   Defines member typedef `type` as the type of the corresponding class
  ///   of the member function pointer.
  ///
  /// - If `F` is class type, class reference type, class pointer type, or
  ///   reference to class pointer type
  ///   Defines member typedef `type` same as the type of the class.
  ///
  /// - For all other cases, no member typedef `type` is provided.
  template <class F, class = void> struct ExtractFunctorTraits {};

  /// Helper type for ExtractFunctorTraits
  template <class F>
  using ExtractFunctorTraits_t = typename ExtractFunctorTraits<F>::type;

  /// Specialization for free function pointer types
  template <class F>
  struct ExtractFunctorTraits<
      F*,
      typename std::enable_if<std::is_function<F>::value>::type> {
    using type = NoObject;
  };

  /// Specialization for member function pointer types
  template <class ReturnType, class C>
  struct ExtractFunctorTraits<ReturnType C::*, void> {
    using type = C;
  };

  /// Specialization for class types
  template <class F>
  struct ExtractFunctorTraits<
      F,
      typename std::enable_if<
          std::is_class<remove_reference_and_pointer_t<F>>::value>::type> {
    using type = remove_reference_and_pointer_t<F>;
  };
} // namespace clad

#endif // FUNCTION_TRAITS
