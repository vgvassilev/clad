//--------------------------------------------------------------------*- C++ -*-
// AutoDiff - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef AUTODIFF_DIFFERENTIATOR
#define AUTODIFF_DIFFERENTIATOR
// We might want to consider using one of C++11 features of std. For now I am 
// sceptical, because they enforce extra conventions that we don't need. Moreover
// by 1.05.2013 it seems that they are not supported on MacOS.

//#include <functional>
//#include <utility>
//void diff(std::function f) {
//void diff(std::mem_fn f) {

#include "BuiltinDerivatives.h"

template<typename ReturnResult, typename... ArgsTypes>
class Function {
public:
  using FunctionType = ReturnResult (*)(ArgsTypes...);
private:
  FunctionType m_Function;
public:
  template<typename... Args>
  Function differentiate(Args&&... args);

  template<typename... Args>
  ReturnResult execute(Args&&... args) {
    return m_Function(args...);
  }
};

// This is the function which will be instantiated with the concrete arguments
// After that our AD library will have all the needed information. For example:
// which is the differentiated function, which is the argument with respect to.
//
// This will be useful in fucture when we are ready to support partial diff.
//

// Note here that best would be to annotate with, eg:
//  __attribute__((annotate("This is our diff that must differentiate"))) {
// However, GCC doesn't support the annotate attribute on a function definition
// and clang's version on MacOS chokes up (with clang's trunk everything seems 
// ok 03.06.2013)

template<typename F, typename... Args, typename... A>
Function<F> diff(F (*f)(Args...), A&&... a) {
  
  //return f(a...);
  return Function<F>();
}
#endif // AUTODIFF_DIFFERENTIATOR
