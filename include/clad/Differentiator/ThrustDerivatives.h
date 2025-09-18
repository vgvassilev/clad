#ifndef CLAD_DIFFERENTIATOR_THRUSTDERIVATIVES_H
#define CLAD_DIFFERENTIATOR_THRUSTDERIVATIVES_H

#include <cstddef>
#include <iterator>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
// NOLINTNEXTLINE(misc-include-cleaner)
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <type_traits>

namespace clad::custom_derivatives::thrust {

template <typename Iterator, typename OutputIterator>
void copy_pullback(Iterator first, Iterator last, OutputIterator result,
                   OutputIterator d_return, Iterator* d_first, Iterator* d_last,
                   OutputIterator* d_result) {
  size_t n = ::thrust::distance(first, last);

  if (n == 0)
    return;

  using ValueConst = typename ::std::iterator_traits<Iterator>::value_type;
  using Value = ::std::remove_const_t<ValueConst>;

  auto d_src_const_ptr = ::thrust::raw_pointer_cast((*d_first).base());
  auto d_src_ptr = const_cast<Value*>(d_src_const_ptr);
  ::thrust::device_ptr<Value> d_src_dev_ptr(d_src_ptr);

  auto d_dst_const_ptr = ::thrust::raw_pointer_cast((*d_result).base());
  auto d_dst_ptr = const_cast<Value*>(d_dst_const_ptr);
  ::thrust::device_ptr<Value> d_dst_dev_ptr(d_dst_ptr);

  struct copy_grad_functor {
    CUDA_HOST_DEVICE void operator()(::thrust::tuple<Value&, Value&> t) const {
      ::thrust::get<0>(t) += ::thrust::get<1>(t);
      ::thrust::get<1>(t) = 0;
    }
  };

  auto iter = ::thrust::make_zip_iterator(
      ::thrust::make_tuple(d_src_dev_ptr, d_dst_dev_ptr));
  ::thrust::for_each(iter, iter + n, copy_grad_functor());
}

template <typename Iterator, typename OutputIterator>
clad::ValueAndAdjoint<OutputIterator, OutputIterator>
copy_reverse_forw(Iterator first, Iterator last, OutputIterator result,
                  Iterator /*dfirst*/, Iterator /*dlast*/,
                  OutputIterator /*dresult*/) {
  return {::thrust::copy(first, last, result), {}};
}

template <typename Iterator, typename T, typename BinaryOp>
void reduce_pullback(Iterator first, Iterator last, T init, BinaryOp op,
                     T d_output, Iterator* d_first, Iterator* d_last, T* d_init,
                     BinaryOp* d_op) {
  size_t n = ::thrust::distance(first, last);

  auto d_first_const_ptr = ::thrust::raw_pointer_cast((*d_first).base());
  auto d_first_ptr = const_cast<T*>(d_first_const_ptr);
  ::thrust::device_ptr<T> d_first_dev_ptr(d_first_ptr);

  if constexpr (::std::is_same_v<BinaryOp, ::thrust::plus<T>>) {
    if (d_init)
      *d_init += d_output;

    struct add_d_output {
      T d_output;
      add_d_output(T d) : d_output(d) {}
      CUDA_HOST_DEVICE void operator()(T& x) const { x += d_output; }
    };

    if (n > 0) {
      ::thrust::for_each(d_first_dev_ptr, d_first_dev_ptr + n,
                         add_d_output(d_output));
    }

  } else if constexpr (::std::is_same_v<BinaryOp, ::thrust::maximum<T>>) {
    auto maxValue = ::thrust::reduce(first, last, init, ::thrust::maximum<T>());
    auto k = (init == maxValue) ? 1 : 0;
    k += ::thrust::count(first, last, maxValue);

    if (k > 0) {
      auto gradient = d_output / k;
      if (d_init && init == maxValue)
        *d_init += gradient;

      struct max_grad_functor {
        T max_val;
        T gradient;
        max_grad_functor(T mv, T g) : max_val(mv), gradient(g) {}
        CUDA_HOST_DEVICE void
        operator()(::thrust::tuple<T&, const T&> t) const {
          if (::thrust::get<1>(t) == max_val)
            ::thrust::get<0>(t) += gradient;
        }
      };

      if (n > 0) {
        auto iter = ::thrust::make_zip_iterator(
            ::thrust::make_tuple(d_first_dev_ptr, first));
        ::thrust::for_each(iter, iter + n,
                           max_grad_functor(maxValue, gradient));
      }
    }

  } else if constexpr (::std::is_same_v<BinaryOp, ::thrust::minimum<T>>) {
    auto minValue = ::thrust::reduce(first, last, init, ::thrust::minimum<T>());
    auto k = (init == minValue) ? 1 : 0;
    k += ::thrust::count(first, last, minValue);

    if (k > 0) {
      auto gradient = d_output / k;
      if (d_init && init == minValue)
        *d_init += gradient;

      struct min_grad_functor {
        T min_val;
        T gradient;
        min_grad_functor(T mv, T g) : min_val(mv), gradient(g) {}
        CUDA_HOST_DEVICE void
        operator()(::thrust::tuple<T&, const T&> t) const {
          if (::thrust::get<1>(t) == min_val)
            ::thrust::get<0>(t) += gradient;
        }
      };

      if (n > 0) {
        auto iter = ::thrust::make_zip_iterator(
            ::thrust::make_tuple(d_first_dev_ptr, first));
        ::thrust::for_each(iter, iter + n,
                           min_grad_functor(minValue, gradient));
      }
    }

  } else if constexpr (::std::is_same_v<BinaryOp, ::thrust::multiplies<T>>) {
    size_t zero_count = ::thrust::count(first, last, 0);
    bool init_is_zero = (init == 0);

    if (zero_count > 1 || (zero_count == 1 && init_is_zero)) {
    } else if (zero_count == 1 && !init_is_zero) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
      struct replace_zero_with_one {
        CUDA_HOST_DEVICE T operator()(const T& x) const {
          return (x == 0) ? 1 : x;
        }
      };
      T non_zero_product = ::thrust::reduce(
          ::thrust::make_transform_iterator(first, replace_zero_with_one()),
          ::thrust::make_transform_iterator(last, replace_zero_with_one()),
          init, ::thrust::multiplies<T>());

      struct single_zero_grad_functor {
        T gradient;
        single_zero_grad_functor(T g) : gradient(g) {}
        CUDA_HOST_DEVICE void
        operator()(::thrust::tuple<T&, const T&> t) const {
          if (::thrust::get<1>(t) == 0)
            ::thrust::get<0>(t) += gradient;
        }
      };
      ::thrust::for_each(::thrust::make_zip_iterator(
                             ::thrust::make_tuple(d_first_dev_ptr, first)),
                         ::thrust::make_zip_iterator(
                             ::thrust::make_tuple(d_first_dev_ptr + n, last)),
                         single_zero_grad_functor(d_output * non_zero_product));

    } else if (zero_count == 0 && init_is_zero) {
      if (d_init) {
        T vector_product =
            ::thrust::reduce(first, last, (T)1, ::thrust::multiplies<T>());
        *d_init += d_output * vector_product;
      }
    } else {
      auto product =
          ::thrust::reduce(first, last, init, ::thrust::multiplies<T>());

      if (d_init)
        *d_init += d_output * product / init;

      struct multiplies_grad_functor {
        T d_output;
        T product;
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
        multiplies_grad_functor(T d, T p) : d_output(d), product(p) {}
        CUDA_HOST_DEVICE T operator()(const T& d_x, const T& x) const {
          return d_x + d_output * product / x;
        }
      };

      if (n > 0) {
        ::thrust::transform(d_first_dev_ptr, d_first_dev_ptr + n, first,
                            d_first_dev_ptr,
                            multiplies_grad_functor(d_output, product));
      }
    }

  } else if constexpr (::std::is_same_v<BinaryOp, ::thrust::bit_and<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::bit_or<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::bit_xor<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::equal_to<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::greater<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::greater_equal<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::less<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::less_equal<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::logical_and<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::logical_or<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::not_equal_to<T>> ||
                       ::std::is_same_v<BinaryOp, ::thrust::modulus<T>>) {
  } else {
    static_assert(::std::is_same_v<T, void>,
                  "This binary operation is not supported by the custom "
                  "reduce_pullback.");
  }
}

template <typename Iterator, typename T>
void reduce_pullback(Iterator first, Iterator last, T init, T d_output,
                     Iterator* d_first, Iterator* d_last, T* d_init) {
  ::thrust::plus<T> op;
  ::thrust::plus<T>* d_op = nullptr;
  reduce_pullback(first, last, init, op, d_output, d_first, d_last, d_init,
                  d_op);
}

template <typename Iterator>
void reduce_pullback(
    Iterator first, Iterator last,
    typename ::std::iterator_traits<Iterator>::value_type d_output,
    Iterator* d_first, Iterator* d_last) {
  using T = typename ::std::iterator_traits<Iterator>::value_type;
  T init = T(0);
  T* d_init = nullptr;
  ::thrust::plus<T> op;
  ::thrust::plus<T>* d_op = nullptr;
  reduce_pullback(first, last, init, op, d_output, d_first, d_last, d_init,
                  d_op);
}

template <typename InputIt, typename OutputIt, typename UnaryOp>
void transform_pullback(InputIt first, InputIt last, OutputIt result,
                        UnaryOp op, OutputIt d_return, InputIt* d_first,
                        InputIt* d_last, OutputIt* d_result, UnaryOp* d_op) {
  size_t n = ::thrust::distance(first, last);

  if (n == 0)
    return;

  using ValueConst = typename ::std::iterator_traits<InputIt>::value_type;
  using Value = ::std::remove_const_t<ValueConst>;

  auto d_src_const_ptr = ::thrust::raw_pointer_cast((*d_first).base());
  auto d_src_ptr = const_cast<Value*>(d_src_const_ptr);
  ::thrust::device_ptr<Value> d_src_dev_ptr(d_src_ptr);

  auto d_dst_const_ptr = ::thrust::raw_pointer_cast((*d_result).base());
  auto d_dst_ptr = const_cast<Value*>(d_dst_const_ptr);
  ::thrust::device_ptr<Value> d_dst_dev_ptr(d_dst_ptr);

  if constexpr (::std::is_same_v<UnaryOp, ::thrust::negate<Value>>) {
    struct grad_functor {
      CUDA_HOST_DEVICE void
      operator()(::thrust::tuple<Value&, Value&> t) const {
        ::thrust::get<0>(t) -= ::thrust::get<1>(t);
        ::thrust::get<1>(t) = 0;
      }
    };
    auto iter = ::thrust::make_zip_iterator(
        ::thrust::make_tuple(d_src_dev_ptr, d_dst_dev_ptr));
    ::thrust::for_each(iter, iter + n, grad_functor());
  } else if constexpr (::std::is_same_v<UnaryOp, ::thrust::identity<Value>>) {
    struct grad_functor {
      CUDA_HOST_DEVICE void
      operator()(::thrust::tuple<Value&, Value&> t) const {
        ::thrust::get<0>(t) += ::thrust::get<1>(t);
        ::thrust::get<1>(t) = 0;
      }
    };
    auto iter = ::thrust::make_zip_iterator(
        ::thrust::make_tuple(d_src_dev_ptr, d_dst_dev_ptr));
    ::thrust::for_each(iter, iter + n, grad_functor());
  } else {
    static_assert(::std::is_same_v<Value, void>,
                  "This unary operation is not supported by the custom "
                  "transform_pullback.");
  }
}

template <typename InputIt, typename OutputIt, typename UnaryOp>
clad::ValueAndAdjoint<OutputIt, OutputIt>
transform_reverse_forw(InputIt first, InputIt last, OutputIt result, UnaryOp op,
                       InputIt /*dfirst*/, InputIt /*dlast*/,
                       OutputIt /*dresult*/, UnaryOp /*dop*/) {
  return {::thrust::transform(first, last, result, op), {}};
}
template <typename InputIt1, typename InputIt2, typename OutputIt,
          typename BinaryOp>
void transform_pullback(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                        OutputIt result, BinaryOp op, OutputIt d_return,
                        InputIt1* d_first1, InputIt1* d_last1,
                        InputIt2* d_first2, OutputIt* d_result,
                        BinaryOp* d_op) {
  size_t n = ::thrust::distance(first1, last1);

  if (n == 0)
    return;

  using ValueConst = typename ::std::iterator_traits<InputIt1>::value_type;
  using Value = ::std::remove_const_t<ValueConst>;

  auto d_first1_const_ptr = ::thrust::raw_pointer_cast((*d_first1).base());
  auto d_first1_ptr = const_cast<Value*>(d_first1_const_ptr);
  ::thrust::device_ptr<Value> d_first1_dev_ptr(d_first1_ptr);

  auto d_first2_const_ptr = ::thrust::raw_pointer_cast((*d_first2).base());
  auto d_first2_ptr = const_cast<Value*>(d_first2_const_ptr);
  ::thrust::device_ptr<Value> d_first2_dev_ptr(d_first2_ptr);

  auto d_result_const_ptr = ::thrust::raw_pointer_cast((*d_result).base());
  auto d_result_ptr = const_cast<Value*>(d_result_const_ptr);
  ::thrust::device_ptr<Value> d_result_dev_ptr(d_result_ptr);

  if constexpr (::std::is_same_v<BinaryOp, ::thrust::plus<Value>>) {
    struct grad_functor {
      CUDA_HOST_DEVICE void
      operator()(::thrust::tuple<Value&, Value&, Value&> t) const {
        ::thrust::get<0>(t) += ::thrust::get<2>(t);
        ::thrust::get<1>(t) += ::thrust::get<2>(t);
        ::thrust::get<2>(t) = 0;
      }
    };
    auto iter = ::thrust::make_zip_iterator(::thrust::make_tuple(
        d_first1_dev_ptr, d_first2_dev_ptr, d_result_dev_ptr));
    ::thrust::for_each(iter, iter + n, grad_functor());
  } else if constexpr (::std::is_same_v<BinaryOp, ::thrust::minus<Value>>) {
    struct grad_functor {
      CUDA_HOST_DEVICE void
      operator()(::thrust::tuple<Value&, Value&, Value&> t) const {
        ::thrust::get<0>(t) += ::thrust::get<2>(t);
        ::thrust::get<1>(t) -= ::thrust::get<2>(t);
        ::thrust::get<2>(t) = 0;
      }
    };
    auto iter = ::thrust::make_zip_iterator(::thrust::make_tuple(
        d_first1_dev_ptr, d_first2_dev_ptr, d_result_dev_ptr));
    ::thrust::for_each(iter, iter + n, grad_functor());
  } else if constexpr (::std::is_same_v<BinaryOp,
                                        ::thrust::multiplies<Value>>) {
    struct grad_functor {
      CUDA_HOST_DEVICE void operator()(
          ::thrust::tuple<Value&, Value&, Value&, const Value&, const Value&> t)
          const {
        // d_x1 += d_y * x2
        ::thrust::get<0>(t) += ::thrust::get<2>(t) * ::thrust::get<4>(t);
        // d_x2 += d_y * x1
        ::thrust::get<1>(t) += ::thrust::get<2>(t) * ::thrust::get<3>(t);
        ::thrust::get<2>(t) = 0;
      }
    };
    auto iter = ::thrust::make_zip_iterator(::thrust::make_tuple(
        d_first1_dev_ptr, d_first2_dev_ptr, d_result_dev_ptr, first1, first2));
    ::thrust::for_each(iter, iter + n, grad_functor());
  } else if constexpr (::std::is_same_v<BinaryOp, ::thrust::divides<Value>>) {
    struct grad_functor {
      CUDA_HOST_DEVICE void operator()(
          ::thrust::tuple<Value&, Value&, Value&, const Value&, const Value&> t)
          const {
        // z = x / y
        const Value& x = ::thrust::get<3>(t);
        const Value& y = ::thrust::get<4>(t);

        // d_x += d_z * (1/y)
        ::thrust::get<0>(t) += ::thrust::get<2>(t) / y;
        // d_y += d_z * (-x / (y*y))
        ::thrust::get<1>(t) -= ::thrust::get<2>(t) * x / (y * y);
        ::thrust::get<2>(t) = 0;
      }
    };
    auto iter = ::thrust::make_zip_iterator(::thrust::make_tuple(
        d_first1_dev_ptr, d_first2_dev_ptr, d_result_dev_ptr, first1, first2));
    ::thrust::for_each(iter, iter + n, grad_functor());
  } else {
    static_assert(::std::is_same_v<Value, void>,
                  "This binary operation is not supported by the custom "
                  "transform_pullback.");
  }
}

template <typename InputIt1, typename InputIt2, typename OutputIt,
          typename BinaryOp>
clad::ValueAndAdjoint<OutputIt, OutputIt>
transform_reverse_forw(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                       OutputIt result, BinaryOp op, InputIt1 d_first1,
                       InputIt1 /*d_last1*/, InputIt2 /*d_first2*/,
                       OutputIt /*d_result*/, BinaryOp /*d_op*/) {
  return {::thrust::transform(first1, last1, first2, result, op), {}};
}

template <typename Iterator1, typename Iterator2, typename T>
void inner_product_pullback(Iterator1 first1, Iterator1 last1, Iterator2 first2,
                            T init, T d_output, Iterator1* d_first1,
                            Iterator1* d_last1, Iterator2* d_first2,
                            T* d_init) {
  if (d_init)
    *d_init += d_output;

  size_t n = ::thrust::distance(first1, last1);

  if (n == 0)
    return;

  auto d_first1_const_ptr = ::thrust::raw_pointer_cast((*d_first1).base());
  auto d_first1_ptr = const_cast<T*>(d_first1_const_ptr);
  ::thrust::device_ptr<T> d_first1_dev_ptr(d_first1_ptr);

  auto d_first2_const_ptr = ::thrust::raw_pointer_cast((*d_first2).base());
  auto d_first2_ptr = const_cast<T*>(d_first2_const_ptr);
  ::thrust::device_ptr<T> d_first2_dev_ptr(d_first2_ptr);

  struct grad_functor {
    T d_output;
    grad_functor(T d) : d_output(d) {}
    CUDA_HOST_DEVICE void
    operator()(::thrust::tuple<T&, T&, const T&, const T&> t) const {
      // d_x1 += d_y * x2
      ::thrust::get<0>(t) += d_output * ::thrust::get<3>(t);
      // d_x2 += d_y * x1
      ::thrust::get<1>(t) += d_output * ::thrust::get<2>(t);
    }
  };

  auto iter = ::thrust::make_zip_iterator(
      ::thrust::make_tuple(d_first1_dev_ptr, d_first2_dev_ptr, first1, first2));
  ::thrust::for_each(iter, iter + n, grad_functor(d_output));
}

template <typename Iterator1, typename Iterator2, typename T,
          typename BinaryOperation1, typename BinaryOperation2>
void inner_product_pullback(Iterator1 first1, Iterator1 last1, Iterator2 first2,
                            T init, BinaryOperation1 op1, BinaryOperation2 op2,
                            T d_output, Iterator1* d_first1, Iterator1* d_last1,
                            Iterator2* d_first2, T* d_init,
                            BinaryOperation1* d_op1, BinaryOperation2* d_op2) {
  if (d_init)
    *d_init += d_output;

  size_t n = ::thrust::distance(first1, last1);

  if (n == 0)
    return;

  auto d_first1_const_ptr = ::thrust::raw_pointer_cast((*d_first1).base());
  auto d_first1_ptr = const_cast<T*>(d_first1_const_ptr);
  ::thrust::device_ptr<T> d_first1_dev_ptr(d_first1_ptr);

  auto d_first2_const_ptr = ::thrust::raw_pointer_cast((*d_first2).base());
  auto d_first2_ptr = const_cast<T*>(d_first2_const_ptr);
  ::thrust::device_ptr<T> d_first2_dev_ptr(d_first2_ptr);

  if constexpr (::std::is_same_v<BinaryOperation1, ::thrust::plus<T>> &&
                ::std::is_same_v<BinaryOperation2, ::thrust::multiplies<T>>) {
    struct grad_functor {
      T d_output;
      grad_functor(T d) : d_output(d) {}
      CUDA_HOST_DEVICE void
      operator()(::thrust::tuple<T&, T&, const T&, const T&> t) const {
        // d_x1 += d_y * x2
        ::thrust::get<0>(t) += d_output * ::thrust::get<3>(t);
        // d_x2 += d_y * x1
        ::thrust::get<1>(t) += d_output * ::thrust::get<2>(t);
      }
    };

    auto iter = ::thrust::make_zip_iterator(::thrust::make_tuple(
        d_first1_dev_ptr, d_first2_dev_ptr, first1, first2));
    ::thrust::for_each(iter, iter + n, grad_functor(d_output));
  } else if constexpr (::std::is_same_v<BinaryOperation1, ::thrust::plus<T>>) {
    if constexpr (::std::is_same_v<BinaryOperation2, ::thrust::plus<T>>) {
      struct grad_functor {
        T d_output;
        grad_functor(T d) : d_output(d) {}
        CUDA_HOST_DEVICE void operator()(::thrust::tuple<T&, T&> t) const {
          ::thrust::get<0>(t) += d_output;
          ::thrust::get<1>(t) += d_output;
        }
      };
      auto iter = ::thrust::make_zip_iterator(
          ::thrust::make_tuple(d_first1_dev_ptr, d_first2_dev_ptr));
      ::thrust::for_each(iter, iter + n, grad_functor(d_output));
    } else if constexpr (::std::is_same_v<BinaryOperation2,
                                          ::thrust::minus<T>>) {
      struct grad_functor {
        T d_output;
        grad_functor(T d) : d_output(d) {}
        CUDA_HOST_DEVICE void operator()(::thrust::tuple<T&, T&> t) const {
          ::thrust::get<0>(t) += d_output;
          ::thrust::get<1>(t) -= d_output;
        }
      };
      auto iter = ::thrust::make_zip_iterator(
          ::thrust::make_tuple(d_first1_dev_ptr, d_first2_dev_ptr));
      ::thrust::for_each(iter, iter + n, grad_functor(d_output));
    } else {
      static_assert(::std::is_same_v<T, void>,
                    "This binary operation is not supported by the custom "
                    "inner_product_pullback.");
    }
  } else {
    static_assert(::std::is_same_v<T, void>,
                  "This binary operation is not supported by the custom "
                  "inner_product_pullback.");
  }
}

template <typename InputIterator, typename UnaryFunction, typename OutputType,
          typename BinaryFunction>
void transform_reduce_pullback(InputIterator first, InputIterator last,
                               UnaryFunction unary_op, OutputType init,
                               BinaryFunction binary_op, OutputType d_output,
                               InputIterator* d_first, InputIterator* d_last,
                               UnaryFunction* d_unary_op, OutputType* d_init,
                               BinaryFunction* d_binary_op) {
  size_t n = ::thrust::distance(first, last);

  if (n == 0) {
    if (d_init)
      *d_init += d_output;
    return;
  }

  using InputType = typename ::std::iterator_traits<InputIterator>::value_type;
  using TransformedType = ::std::decay_t<
      typename ::std::invoke_result<UnaryFunction, InputType>::type>;

  // 1. Perform the forward transform to get intermediate values.
  ::thrust::device_vector<TransformedType> transformed_values(n);
  ::thrust::transform(first, last, transformed_values.begin(), unary_op);

  // 2. Compute gradients for the intermediate transformed values by calling
  // reduce_pullback.
  ::thrust::device_vector<TransformedType> d_transformed_values(n);
  auto d_transformed_begin = d_transformed_values.begin();
  auto d_transformed_end_dummy = d_transformed_values.end();

  reduce_pullback(transformed_values.begin(), transformed_values.end(), init,
                  binary_op, d_output, &d_transformed_begin,
                  &d_transformed_end_dummy, d_init, d_binary_op);

  // 3. Propagate gradients from the transformed values back to the original
  // input.
  ::thrust::device_vector<TransformedType> d_result_dummy(n);
  auto d_transformed_values_it = d_transformed_values.begin();
  transform_pullback(first, last, transformed_values.begin(), unary_op,
                     d_result_dummy.begin(), // d_return dummy
                     d_first, d_last, &d_transformed_values_it, d_unary_op);
}

} // namespace clad::custom_derivatives::thrust

#endif // CLAD_DIFFERENTIATOR_THRUSTDERIVATIVES_H
