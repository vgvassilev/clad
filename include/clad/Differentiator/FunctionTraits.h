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

  // ExtractDerivedFnTraits is used to deduce type of the derived functions 
  // derived using reverse, hessian and jacobian differentiation modes
  // It SHOULD NOT be used to get traits of derived functions derived using
  // forward differentiation mode
  template<class ReturnType, bool isDerivedFn>
  struct ExtractDerivedFnTraits {};
  template<class T, bool isDerivedFn>
  using ExtractDerivedFnTraits_t = typename ExtractDerivedFnTraits<T, isDerivedFn>::type;

  // specializations for non-member functions pointer types
  template <class ReturnType, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (*)(Args...), false> {
    using type = void (*)(Args..., ReturnType*);
  };

  template <class ReturnType, class... Args>
  struct ExtractDerivedFnTraits<ReturnType (*)(Args...), true> {
    using type = ReturnType (*)(Args...);
  };

  // specializations for member functions pointer types with no cv-qualifiers
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...), false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };

  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...), true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  
  // specializations for member functions pointer types with only cv-qualifiers
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) volatile, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const volatile, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) volatile, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const volatile, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };

  // specializations for member functions pointer types with 
  // reference qualifiers and with and without cv-qualifiers

  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) &, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const &, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) volatile &, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const volatile &, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) &&, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const &&, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) volatile &&, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const volatile &&, true> { 
    using type = ReturnType (C::*)(Args...); 
  };

  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) &, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const &, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) volatile &, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const volatile &, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) &&, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const &&, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) volatile &&, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const volatile &&, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };

  // specializations for noexcept member functions
  #if __cpp_noexcept_function_type > 0

  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) noexcept, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const noexcept, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) volatile noexcept, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const volatile noexcept, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) & noexcept, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const & noexcept, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) volatile & noexcept, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const volatile & noexcept, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) && noexcept, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const && noexcept, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) volatile && noexcept, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const volatile && noexcept, true> { 
    using type = ReturnType (C::*)(Args...); 
  };
  
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) noexcept, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const noexcept, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) volatile noexcept, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const volatile noexcept, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) & noexcept, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const & noexcept, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) volatile & noexcept, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const volatile & noexcept, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) && noexcept, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const && noexcept, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) volatile && noexcept, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  template <class ReturnType, class C, class... Args> 
  struct ExtractDerivedFnTraits<ReturnType (C::*)(Args...) const volatile && noexcept, false> { 
    using type = void (C::*)(Args..., ReturnType*); 
  };
  #endif
} // namespace clad

#endif // FUNCTION_TRAITS