#ifndef FUNCTION_TRAITS
#define FUNCTION_TRAITS

namespace clad {

  // Trait class to deduce return type of function(both member and non-member) at commpile time
  // Only function pointer types are supported by this trait class
  template <class F> 
  struct return_type {};
  template <class F> 
  using return_type_t = typename return_type<F>::type;

  // specializations for non-member functions pointer types
  template <class ReturnType, class... Args> 
  struct return_type<ReturnType (*)(Args...)> {
    using type = ReturnType;
  };
  template <class ReturnType, class... Args> 
  struct return_type<ReturnType (*)(Args..., ...)> {
    using type = ReturnType;
  };

  // specializations for member functions pointer types with no qualifiers
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...)> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...)> { 
    using type = ReturnType; 
  };

  // specializations for member functions pointer type with only cv-qualifiers
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) const> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) const> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) volatile> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) volatile> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) const volatile> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) const volatile> { 
    using type = ReturnType; 
  };

  // specializations for member functions pointer types with 
  // reference qualifiers and with and without cv-qualifiers
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) &> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) &> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) const &> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) const &> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) volatile &> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) volatile &> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) const volatile &> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) const volatile &> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) &&> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) &&> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) const &&> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) const &&> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) volatile &&> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) volatile &&> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) const volatile &&> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) const volatile &&> { 
    using type = ReturnType; 
  };

  // specializations for noexcept member functions
  #if __cpp_noexcept_function_type > 0
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) const noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) const noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) volatile noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) volatile noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) const volatile noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) const volatile noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) & noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) & noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) const & noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) const & noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) volatile & noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) volatile & noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) const volatile & noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) const volatile & noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) && noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) && noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) const && noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) const && noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) volatile && noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) volatile && noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args...) const volatile && noexcept> { 
    using type = ReturnType; 
  };
  template <class ReturnType, class C, class... Args> 
  struct return_type<ReturnType (C::*)(Args..., ...) const volatile && noexcept> { 
    using type = ReturnType; 
  };
  #endif

#define REM_CTOR(...) __VA_ARGS__

  // Setup for DropArgs
  template <typename... T> struct list {};

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

  template <std::size_t, typename T> struct DropArgs;

  // Returns the Args in the function F that come after the Nth arg
  template <std::size_t N, typename F>
  using DropArgs_t = typename DropArgs<N, F>::type;

  template <std::size_t N, typename R, typename... Args>
  struct DropArgs<N, R (*)(Args...)> {
    using type = Drop_t<N, Args...>;
  };

  template <std::size_t N, typename R, typename... Args>
  struct DropArgs<N, R (*)(Args..., ...)> {
    using type = Drop_t<N, Args...>;
  };

  /// These macro expansions are used to cover all possible cases of
  /// qualifiers in member functions. The need to be read from bottom to top
  /// Starting from the use of AddCON, the call to which is used to pass the
  /// case about with and without C-style varargs, then as the macro name says
  /// it adds cases of const qualifier. The AddVOL and AddREF macro similarly
  /// add cases for volatile qualifier and reference respectively. The AddNOEX
  /// adds cases for noexcept qualifier only if it is supported and finally
  /// AddSPECS declares the function with all the cases
  // DropArgs specializations for member function pointer types
#define DropArgs_AddSPECS(var, con, vol, ref, noex)                            \
  template <std::size_t N, typename R, typename C, typename... Args>           \
  struct DropArgs<N, R (C::*)(Args... REM_CTOR var) con vol ref noex> {        \
    using type = Drop_t<N, Args...>;                                           \
  };

#if __cpp_noexcept_function_type > 0
#define DropArgs_AddNOEX(var, con, vol, ref)                                   \
  DropArgs_AddSPECS(var, con, vol, ref, )                                      \
      DropArgs_AddSPECS(var, con, vol, ref, noexcept)
#else
#define DropArgs_AddNOEX(var, con, vol, ref)                                   \
  DropArgs_AddSPECS(var, con, vol, ref, )
#endif

#define DropArgs_AddREF(var, con, vol)                                         \
  DropArgs_AddNOEX(var, con, vol, ) DropArgs_AddNOEX(var, con, vol, &)         \
      DropArgs_AddNOEX(var, con, vol, &&)

#define DropArgs_AddVOL(var, con)                                              \
  DropArgs_AddREF(var, con, ) DropArgs_AddREF(var, con, volatile)

#define DropArgs_AddCON(var) DropArgs_AddVOL(var, ) DropArgs_AddVOL(var, const)

  DropArgs_AddCON(())
      DropArgs_AddCON((, ...)) // Declares all the specializations

      template <class T, class R>
      struct OutputParamType {
    using type = R*;
  };

  template <class T, class R>
  using OutputParamType_t = typename OutputParamType<T, R>::type;

  template <class T> struct GradientDerivedFnTraits {};

  // GradientDerivedFnTraits is used to deduce type of the derived functions
  // derived using reverse modes
  template <class T>
  using GradientDerivedFnTraits_t = typename GradientDerivedFnTraits<T>::type;

  // GradientDerivedFnTraits specializations for pure function pointer types
  template <class ReturnType, class... Args>
  struct GradientDerivedFnTraits<ReturnType (*)(Args...)> {
    using type = void (*)(Args..., OutputParamType_t<Args, ReturnType>...);
  };

  // GradientDerivedFnTraits specializations for member function pointer types
#define GradientDerivedFnTraits_AddSPECS(var, cv, vol, ref, noex)              \
  template <typename R, typename C, typename... Args>                          \
  struct GradientDerivedFnTraits<R (C::*)(Args...) cv vol ref noex> {          \
    using type = void (C::*)(Args...,                                          \
                             OutputParamType_t<Args, R>...) cv vol ref noex;   \
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

  GradientDerivedFnTraits_AddCON(()) // Declares all the specializations

      template <class... Args>
      struct SelectLast;

  template <class... Args>
  using SelectLast_t = typename SelectLast<Args...>::type;

  template <class T> struct SelectLast<T> { using type = T; };

  template <class T, class... Args> struct SelectLast<T, Args...> {
    using type = typename SelectLast<Args...>::type;
  };

  template <class T> struct JacobianDerivedFnTraits {};

  // JacobianDerivedFnTraits is used to deduce type of the derived functions
  // derived using jacobian mode
  template <class T>
  using JacobianDerivedFnTraits_t = typename JacobianDerivedFnTraits<T>::type;

  // JacobianDerivedFnTraits specializations for pure function pointer types
  template <class ReturnType, class... Args>
  struct JacobianDerivedFnTraits<ReturnType (*)(Args...)> {
    using type = void (*)(Args..., SelectLast_t<Args...>);
  };

  // JacobianDerivedFnTraits specializations for member function pointer types
#define JacobianDerivedFnTraits_AddSPECS(var, cv, vol, ref, noex)              \
  template <typename R, typename C, typename... Args>                          \
  struct JacobianDerivedFnTraits<R (C::*)(Args...) cv vol ref noex> {          \
    using type = void (C::*)(Args..., SelectLast_t<Args...>) cv vol ref noex;  \
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

  JacobianDerivedFnTraits_AddCON(()) // Declares all the specializations

      // ExtractDerivedFnTraits is used to deduce type of the derived functions
      // derived using hessian and jacobian differentiation modes
      // It SHOULD NOT be used to get traits of derived functions derived using
      // forward or reverse differentiation mode
      template <class ReturnType>
      struct ExtractDerivedFnTraits {};
  template<class T>
  using ExtractDerivedFnTraits_t = typename ExtractDerivedFnTraits<T>::type;

  // specializations for non-member functions pointer types
  template <class ReturnType,class... Args>
  struct ExtractDerivedFnTraits<ReturnType (*)(Args...)> {
    using type = void (*)(Args..., ReturnType*);
  };

  // specializations for member functions pointer types with no cv-qualifiers
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...)> {
    using type = void (C::*)(Args..., ReturnType*);
  };

  // specializations for member functions pointer types with only cv-qualifiers
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const> {
    using type = void (C::*)(Args..., ReturnType*) const;
  };
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) volatile> {
    using type = void (C::*)(Args..., ReturnType*) volatile;
  };
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const volatile> {
    using type = void (C::*)(Args..., ReturnType*) const volatile;
  };

  // specializations for member functions pointer types with 
  // reference qualifiers and with and without cv-qualifiers
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) &> {
    using type = void (C::*)(Args..., ReturnType*) &;
  };
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const &> {
    using type = void (C::*)(Args..., ReturnType*) const &;
  };
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) volatile &> {
    using type = void (C::*)(Args..., ReturnType*) volatile &;
  };
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const volatile &> {
    using type = void (C::*)(Args..., ReturnType*) const volatile &;
  };
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) &&> {
    using type = void (C::*)(Args..., ReturnType*) &&;
  };
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const &&> {
    using type = void (C::*)(Args..., ReturnType*) const &&;
  };
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) volatile &&> {
    using type = void (C::*)(Args..., ReturnType*) volatile &&;
  };
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const volatile &&> {
    using type = void (C::*)(Args..., ReturnType*) const volatile &&;
  };

  // specializations for noexcept member functions
  #if __cpp_noexcept_function_type > 0
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) noexcept> {
    using type = void (C::*)(Args..., ReturnType*) noexcept;
  };
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const noexcept> {
    using type = void (C::*)(Args..., ReturnType*) const noexcept;
  };
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) volatile noexcept> {
    using type = void (C::*)(Args..., ReturnType*) volatile noexcept;
  };
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...)
                                    const volatile noexcept> {
    using type = void (C::*)(Args..., ReturnType*) const volatile noexcept;
  };
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) & noexcept> {
    using type = void (C::*)(Args..., ReturnType*) & noexcept;
  };
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const & noexcept> {
    using type = void (C::*)(Args..., ReturnType*) const & noexcept;
  };
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) volatile& noexcept> {
    using type = void (C::*)(Args..., ReturnType*) volatile& noexcept;
  };

  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...)
                                    const volatile& noexcept> {
    using type = void (C::*)(Args..., ReturnType*) const volatile& noexcept;
  };
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) && noexcept> {
    using type = void (C::*)(Args..., ReturnType*) && noexcept;
  };
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const && noexcept> {
    using type = void (C::*)(Args..., ReturnType*) const && noexcept;
  };
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(
      Args...) volatile&& noexcept> {
    using type = void (C::*)(Args..., ReturnType*) volatile&& noexcept;
  };
  template <class ReturnType, class C, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...)
                                    const volatile&& noexcept> {
    using type = void (C::*)(Args..., ReturnType*) const volatile&& noexcept;
  };
#endif
} // namespace clad

#endif // FUNCTION_TRAITS