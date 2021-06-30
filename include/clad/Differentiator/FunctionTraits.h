#ifndef FUNCTION_TRAITS
#define FUNCTION_TRAITS
#include <type_traits>

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
  template<class ReturnType>
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

  /// Check whether class 'C` defines call operator. Provides the member
  /// constant `value` which is equal to true, if class defines call operator.
  /// Otherwise `value` is equal to false.
  template <typename C, typename = void>
  struct has_call_operator : std::false_type {};

  template <typename C>
  struct has_call_operator<
      C,
      typename std::enable_if<(sizeof(&C::operator()) > 0)>::type>
      : std::true_type {};

  /// Compute type of derived function of function, method or functor when
  /// differentiated using forward differentiation mode (`clad::differentiate`).
  /// Computed type is provided as member typedef `type`.
  ///
  /// More precisely, this type trait behaves as following:
  ///
  /// - If `F` is a function pointer type
  ///   Defines member typedef `type` same as the type of the function pointer.
  ///
  /// - If `F` is a member function pointer type
  ///   Defines member typedef `type` same as the type of the member function
  /// pointer.
  ///
  /// - If `F` is class type, class reference type, class pointer type, or
  ///   reference to class pointer type.
  ///   Defines member typedef `type` same as the type of the overloaded call
  ///   operator member function of the class.
  ///
  /// - For all other cases, no member typedef `type` is provided.
  template <class F, class = void> struct ExtractDerivedFnTraitsForwMode {};

  /// Helper type for ExtractDerivedFnTraitsForwMode
  template <class F>
  using ExtractDerivedFnTraitsForwMode_t =
      typename ExtractDerivedFnTraitsForwMode<F>::type;

  /// Specialization for free function pointer type
  template <class F>
  struct ExtractDerivedFnTraitsForwMode<
      F*,
      typename std::enable_if<std::is_function<F>::value>::type> {
    using type = remove_reference_and_pointer_t<F>*;
  };

  /// Specialization for member function pointer type
  template <class F>
  struct ExtractDerivedFnTraitsForwMode<
      F,
      typename std::enable_if<
          std::is_member_function_pointer<F>::value>::type> {
    using type = typename std::decay<F>::type;
  };

  /// Specialization for class types
  template <class F>
  struct ExtractDerivedFnTraitsForwMode<
      F,
      typename std::enable_if<
          std::is_class<remove_reference_and_pointer_t<F>>::value>::type> {
    using ClassType =
        typename std::decay<remove_reference_and_pointer_t<F>>::type;
    static_assert(
        has_call_operator<ClassType>::value,
        "Passed object do not have overloaded call operator member function");
    using type = decltype(&ClassType::operator());
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
  ///   Defines member typedef 'type` same as the type of the class.
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