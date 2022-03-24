#ifndef TEST_TEST_UTILS_H
#define TEST_TEST_UTILS_H
#include "clad/Differentiator/ArrayRef.h"

#include <cstdio>
#include <type_traits>

namespace test_utils {

template<typename T>
void print(T t) {
  fprintf(stderr, "Print method not defined for type: %s", typeid(t).name());
}

void print(const char* s) { printf("%s", s); }

void print(double d) { printf("%.2f", d); }

void print(int i) { printf("%d", i); }

void print() {}

template <typename T> void print(clad::array_ref<T> arr) {
  for (std::size_t i = 0; i < arr.size(); ++i) {
    print(arr[i]);
    if (i != arr.size() - 1)
      printf(", ");
  }
}

template <typename Arg, typename... Args> void print(Arg* arg, Args... args);

// Prints comma separated list of all the arguments values.
template <typename Arg, typename... Args,
          class = typename std::enable_if<!std::is_pointer<Arg>::value>::type>
void print(Arg arg, Args... args) {
  print(arg);
  if (sizeof...(Args) > 0)
    printf(", ");
  print(args...);
}

template <typename Arg, typename... Args> void print(Arg* arg, Args... args) {
  print(*arg);
  if (sizeof...(Args) > 0)
    printf(", ");
  print(args...);
}

// Prints comma separated list of all the arguments values enclosed in curly
// braces and terminated by a new line.
template <typename... Args> void display(Args... args) {
  printf("{");
  print(args...);
  printf("}\n");
}

template <std::size_t...> struct index_pack {};

template <std::size_t l, std::size_t r, std::size_t... S>
struct GenerateRange : GenerateRange<l + 1, r, S..., l> {};

template <std::size_t l, std::size_t... S> struct GenerateRange<l, l, S...> {
  using type = index_pack<S..., l>;
};

void reset() {}

template <typename Arg,
          class = typename std::enable_if<!std::is_pointer<Arg>::value>::type>
void reset(Arg& arg) {
  arg = Arg();
}

template <typename Arg> void reset(Arg* arg) { *arg = Arg(); }

template <typename T> void reset(clad::array_ref<T> arr) {
  for (std::size_t i = 0; i < arr.size(); ++i)
    reset(arr[i]);
}

template <typename Arg, typename... Args> void reset(Arg& arg, Args&... args) {
  reset(arg);
  reset(args...);
}

template <class CF, std::size_t... S, class... Args>
void run_gradient_impl(CF cf, index_pack<S...> s, Args&&... args) {
  std::tuple<Args...> t = {args...};
  reset(std::get<S>(t)...);
  cf.execute(args...);
  display(std::get<S>(t)...);
}

template <std::size_t NumOfDerivativeArgs, class CF, class... Args>
void run_gradient(CF cf, Args&&... args) {
  using DerivativeArgsRange =
      typename GenerateRange<sizeof...(Args) - NumOfDerivativeArgs,
                             sizeof...(Args) - 1>::type;
  run_gradient_impl(cf, DerivativeArgsRange(), std::forward<Args>(args)...);
}

template <class CF, class... Args>
void run_differentiate(CF cf, Args&&... args) {
  display(cf.execute(std::forward<Args>(args)...));
}

#define INIT_GRADIENT_ALL(fn) auto fn##_grad = clad::gradient(fn);

#define INIT_DIFFERENTIATE(fn, ...)                                            \
  auto fn##_diff = clad::differentiate(fn, __VA_ARGS__);

#define INIT_GRADIENT_SPECIFIC(fn, args)                                       \
  auto fn##_grad = clad::gradient(fn, args);

#define GET_MACRO(_1, _2, MACRO, ...) MACRO

#define INIT_GRADIENT(...)                                                     \
  GET_MACRO(__VA_ARGS__, INIT_GRADIENT_SPECIFIC, INIT_GRADIENT_ALL)            \
  (__VA_ARGS__)

#define TEST_GRADIENT(fn, numOfDerivativeArgs, ...)                            \
  test_utils::run_gradient<numOfDerivativeArgs>(fn##_grad, __VA_ARGS__);

#define TEST_DIFFERENTIATE(fn, ...)                                            \
  test_utils::run_differentiate(fn##_diff, __VA_ARGS__);

#endif
}