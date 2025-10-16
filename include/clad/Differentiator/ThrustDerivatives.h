#ifndef CLAD_DIFFERENTIATOR_THRUSTDERIVATIVES_H
#define CLAD_DIFFERENTIATOR_THRUSTDERIVATIVES_H

#include <cstddef>
#include <iterator>
#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h> // NOLINT(misc-include-cleaner)
// NOLINTNEXTLINE(misc-include-cleaner)
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <type_traits>

namespace clad::custom_derivatives::thrust {

namespace detail {
// Detection of functor operator_call_pullback for unary and binary ops
template <typename, typename, typename = void>
struct has_unary_operator_call_pullback : ::std::false_type {};

template <typename Op, typename T>
struct has_unary_operator_call_pullback<
    Op, T,
    ::std::void_t<decltype(::std::declval<Op&>().operator_call_pullback(
        ::std::declval<T>(),                   // x
        ::std::declval<T>(),                   // d_y
        (::std::add_pointer_t<Op>)(nullptr),   // d_op
        (::std::add_pointer_t<T>)(nullptr)))>> // d_x
    : ::std::true_type {};

template <typename, typename, typename = void>
struct has_binary_operator_call_pullback : ::std::false_type {};

template <typename Op, typename T>
struct has_binary_operator_call_pullback<
    Op, T,
    ::std::void_t<decltype(::std::declval<Op&>().operator_call_pullback(
        ::std::declval<T>(),                   // x1
        ::std::declval<T>(),                   // x2
        ::std::declval<T>(),                   // d_y
        (::std::add_pointer_t<Op>)(nullptr),   // d_op
        (::std::add_pointer_t<T>)(nullptr),    // d_x1
        (::std::add_pointer_t<T>)(nullptr)))>> // d_x2
    : ::std::true_type {};
} // namespace detail

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

template <typename Iterator, typename T, typename BinaryOp>
T reduce_reverse_forw(Iterator first, Iterator last, T init, BinaryOp op,
                      Iterator /*dfirst*/, Iterator /*dlast*/, T /*dinit*/,
                      BinaryOp /*dop*/) {
  return ::thrust::reduce(first, last, init, op);
}

template <typename Iterator, typename T>
T reduce_reverse_forw(Iterator first, Iterator last, T init,
                      Iterator /*dfirst*/, Iterator /*dlast*/, T /*dinit*/) {
  return ::thrust::reduce(first, last, init);
}

template <typename Iterator>
typename ::std::iterator_traits<Iterator>::value_type
reduce_reverse_forw(Iterator first, Iterator last, Iterator /*dfirst*/,
                    Iterator /*dlast*/) {
  return ::thrust::reduce(first, last);
}

template <typename InputIt, typename OutputIt>
void inclusive_scan_pullback(InputIt first, InputIt last, OutputIt result,
                             OutputIt d_return, InputIt* d_first,
                             InputIt* d_last, OutputIt* d_result) {
  using Value = typename ::std::iterator_traits<InputIt>::value_type;
  ::thrust::plus<Value> op;
  ::thrust::plus<Value>* d_op = nullptr;
  inclusive_scan_pullback(first, last, result, op, d_return, d_first, d_last,
                          d_result, d_op);
}

template <typename InputIt, typename OutputIt, typename BinaryOp>
void inclusive_scan_pullback(InputIt first, InputIt last, OutputIt result,
                             BinaryOp op, OutputIt d_return, InputIt* d_first,
                             InputIt* d_last, OutputIt* d_result,
                             BinaryOp* d_op) {
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

  if constexpr (::std::is_same_v<BinaryOp, ::thrust::plus<Value>>) {
    ::thrust::device_vector<Value> suffix_sums(n);

    ::thrust::inclusive_scan(::thrust::make_reverse_iterator(d_dst_dev_ptr + n),
                             ::thrust::make_reverse_iterator(d_dst_dev_ptr),
                             suffix_sums.begin(), op);

    ::thrust::transform(d_src_dev_ptr, d_src_dev_ptr + n,
                        ::thrust::make_reverse_iterator(suffix_sums.end()),
                        d_src_dev_ptr, ::thrust::plus<Value>());

    ::thrust::fill(d_dst_dev_ptr, d_dst_dev_ptr + n, Value(0));

  } else {
    static_assert(::std::is_same_v<Value, void>,
                  "This binary operation is not supported by the custom "
                  "inclusive_scan_pullback.");
  }
}

template <typename InputIt, typename OutputIt>
clad::ValueAndAdjoint<OutputIt, OutputIt>
inclusive_scan_reverse_forw(InputIt first, InputIt last, OutputIt result,
                            InputIt, InputIt, OutputIt) {
  using Value = typename ::std::iterator_traits<InputIt>::value_type;
  return {
      ::thrust::inclusive_scan(first, last, result, ::thrust::plus<Value>()),
      {}};
}

template <typename InputIt, typename OutputIt, typename BinaryOp>
clad::ValueAndAdjoint<OutputIt, OutputIt>
inclusive_scan_reverse_forw(InputIt first, InputIt last, OutputIt result,
                            BinaryOp op, InputIt, InputIt, OutputIt, BinaryOp) {
  return {::thrust::inclusive_scan(first, last, result, op), {}};
}

template <typename InputIt, typename OutputIt, typename T>
void exclusive_scan_pullback(InputIt first, InputIt last, OutputIt result,
                             T init, OutputIt d_return, InputIt* d_first,
                             InputIt* d_last, OutputIt* d_result, T* d_init) {
  using Value = typename ::std::iterator_traits<InputIt>::value_type;
  ::thrust::plus<Value> op;
  ::thrust::plus<Value>* d_op = nullptr;
  exclusive_scan_pullback(first, last, result, init, op, d_return, d_first,
                          d_last, d_result, d_init, d_op);
}

template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
void exclusive_scan_pullback(InputIt first, InputIt last, OutputIt result,
                             T init, BinaryOp op, OutputIt d_return,
                             InputIt* d_first, InputIt* d_last,
                             OutputIt* d_result, T* d_init, BinaryOp* d_op) {
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

  if constexpr (::std::is_same_v<BinaryOp, ::thrust::plus<Value>>) {
    if (d_init) {
      *d_init +=
          ::thrust::reduce(d_dst_dev_ptr, d_dst_dev_ptr + n, Value(0), op);
    }

    ::thrust::device_vector<Value> suffix_sums(n);

    ::thrust::exclusive_scan(::thrust::make_reverse_iterator(d_dst_dev_ptr + n),
                             ::thrust::make_reverse_iterator(d_dst_dev_ptr),
                             suffix_sums.begin(), Value(0), op);

    ::thrust::transform(d_src_dev_ptr, d_src_dev_ptr + n,
                        ::thrust::make_reverse_iterator(suffix_sums.end()),
                        d_src_dev_ptr, ::thrust::plus<Value>());

    ::thrust::fill(d_dst_dev_ptr, d_dst_dev_ptr + n, Value(0));
  } else {
    static_assert(::std::is_same_v<Value, void>,
                  "This binary operation is not supported by the custom "
                  "exclusive_scan_pullback.");
  }
}

template <typename InputIt, typename OutputIt, typename T>
clad::ValueAndAdjoint<OutputIt, OutputIt>
exclusive_scan_reverse_forw(InputIt first, InputIt last, OutputIt result,
                            T init, InputIt, InputIt, OutputIt, T) {
  using Value = typename ::std::iterator_traits<InputIt>::value_type;
  return {::thrust::exclusive_scan(first, last, result, init,
                                   ::thrust::plus<Value>()),
          {}};
}

template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
clad::ValueAndAdjoint<OutputIt, OutputIt>
exclusive_scan_reverse_forw(InputIt first, InputIt last, OutputIt result,
                            T init, BinaryOp op, InputIt, InputIt, OutputIt, T,
                            BinaryOp) {
  return {::thrust::exclusive_scan(first, last, result, init, op), {}};
}

template <typename KeyIt, typename ValIt, typename OutputIt>
void inclusive_scan_by_key_pullback(KeyIt keys_first, KeyIt keys_last,
                                    ValIt values_first, OutputIt result,
                                    OutputIt /*d_return*/, KeyIt* d_keys_first,
                                    KeyIt* d_keys_last, ValIt* d_values_first,
                                    OutputIt* d_result) {
  using Key = typename ::std::iterator_traits<KeyIt>::value_type;
  using ValueConst = typename ::std::iterator_traits<ValIt>::value_type;
  using Value = ::std::remove_const_t<ValueConst>;
  ::thrust::equal_to<Key> pred;
  ::thrust::plus<Value> op;
  ::thrust::equal_to<Key>* d_pred = nullptr;
  ::thrust::plus<Value>* d_op = nullptr;
  inclusive_scan_by_key_pullback(keys_first, keys_last, values_first, result,
                                 pred, op, {}, d_keys_first, d_keys_last,
                                 d_values_first, d_result, d_pred, d_op);
}

template <typename KeyIt, typename ValIt, typename OutputIt, typename KeyEqual,
          typename BinaryOp>
void inclusive_scan_by_key_pullback(KeyIt keys_first, KeyIt keys_last,
                                    ValIt values_first, OutputIt result,
                                    KeyEqual /*pred*/, BinaryOp op,
                                    OutputIt /*d_return*/, KeyIt* d_keys_first,
                                    KeyIt* d_keys_last, ValIt* d_values_first,
                                    OutputIt* d_result, KeyEqual* /*d_pred*/,
                                    BinaryOp* /*d_op*/) {
  size_t n = ::thrust::distance(keys_first, keys_last);
  if (n == 0)
    return;

  using ValueConst = typename ::std::iterator_traits<ValIt>::value_type;
  using Value = ::std::remove_const_t<ValueConst>;
  using Key = typename ::std::iterator_traits<KeyIt>::value_type;

  auto d_src_const_ptr = ::thrust::raw_pointer_cast((*d_values_first).base());
  auto d_src_ptr = const_cast<Value*>(d_src_const_ptr);
  ::thrust::device_ptr<Value> d_src_dev_ptr(d_src_ptr);

  auto d_dst_const_ptr = ::thrust::raw_pointer_cast((*d_result).base());
  auto d_dst_ptr = const_cast<Value*>(d_dst_const_ptr);
  ::thrust::device_ptr<Value> d_dst_dev_ptr(d_dst_ptr);

  if constexpr (::std::is_same_v<BinaryOp, ::thrust::plus<Value>> &&
                ::std::is_same_v<KeyEqual, ::thrust::equal_to<Key>>) {
    ::thrust::device_vector<Value> suffix_sums(n);

    auto rev_keys_begin = ::thrust::make_reverse_iterator(keys_last);
    auto rev_keys_end = ::thrust::make_reverse_iterator(keys_first);
    auto rev_dy_begin = ::thrust::make_reverse_iterator(d_dst_dev_ptr + n);
    auto rev_dy_end = ::thrust::make_reverse_iterator(d_dst_dev_ptr);

    ::thrust::inclusive_scan_by_key(rev_keys_begin, rev_keys_end, rev_dy_begin,
                                    suffix_sums.begin(),
                                    ::thrust::equal_to<Key>(), op);

    ::thrust::transform(d_src_dev_ptr, d_src_dev_ptr + n,
                        ::thrust::make_reverse_iterator(suffix_sums.end()),
                        d_src_dev_ptr, ::thrust::plus<Value>());

    ::thrust::fill(d_dst_dev_ptr, d_dst_dev_ptr + n, Value(0));
  } else {
    static_assert(::std::is_same_v<Value, void>,
                  "This predicate/operation is not supported by the custom "
                  "inclusive_scan_by_key_pullback.");
  }
}

template <typename KeyIt, typename ValIt, typename OutputIt>
clad::ValueAndAdjoint<OutputIt, OutputIt>
inclusive_scan_by_key_reverse_forw(KeyIt keys_first, KeyIt keys_last,
                                   ValIt values_first, OutputIt result, KeyIt,
                                   KeyIt, ValIt, OutputIt) {
  using Key = typename ::std::iterator_traits<KeyIt>::value_type;
  using Value = typename ::std::iterator_traits<ValIt>::value_type;
  return {::thrust::inclusive_scan_by_key(keys_first, keys_last, values_first,
                                          result, ::thrust::equal_to<Key>(),
                                          ::thrust::plus<Value>()),
          {}};
}

template <typename KeyIt, typename ValIt, typename OutputIt, typename KeyEqual,
          typename BinaryOp>
clad::ValueAndAdjoint<OutputIt, OutputIt>
inclusive_scan_by_key_reverse_forw(KeyIt keys_first, KeyIt keys_last,
                                   ValIt values_first, OutputIt result,
                                   KeyEqual pred, BinaryOp op, KeyIt, KeyIt,
                                   ValIt, OutputIt, KeyEqual, BinaryOp) {
  return {::thrust::inclusive_scan_by_key(keys_first, keys_last, values_first,
                                          result, pred, op),
          {}};
}

template <typename KeyIt, typename ValIt, typename OutputIt, typename T>
void exclusive_scan_by_key_pullback(KeyIt keys_first, KeyIt keys_last,
                                    ValIt values_first, OutputIt result, T init,
                                    OutputIt /*d_return*/, KeyIt* d_keys_first,
                                    KeyIt* d_keys_last, ValIt* d_values_first,
                                    OutputIt* d_result, T* d_init) {
  using Key = typename ::std::iterator_traits<KeyIt>::value_type;
  using ValueConst = typename ::std::iterator_traits<ValIt>::value_type;
  using Value = ::std::remove_const_t<ValueConst>;
  ::thrust::equal_to<Key> pred;
  ::thrust::plus<Value> op;
  ::thrust::equal_to<Key>* d_pred = nullptr;
  ::thrust::plus<Value>* d_op = nullptr;
  exclusive_scan_by_key_pullback(keys_first, keys_last, values_first, result,
                                 init, pred, op, {}, d_keys_first, d_keys_last,
                                 d_values_first, d_result, d_init, d_pred,
                                 d_op);
}

template <typename KeyIt, typename ValIt, typename OutputIt, typename T,
          typename KeyEqual, typename BinaryOp>
void exclusive_scan_by_key_pullback(KeyIt keys_first, KeyIt keys_last,
                                    ValIt values_first, OutputIt result, T init,
                                    KeyEqual /*pred*/, BinaryOp op,
                                    OutputIt /*d_return*/, KeyIt* d_keys_first,
                                    KeyIt* d_keys_last, ValIt* d_values_first,
                                    OutputIt* d_result, T* d_init,
                                    KeyEqual* /*d_pred*/, BinaryOp* /*d_op*/) {
  size_t n = ::thrust::distance(keys_first, keys_last);
  if (n == 0) {
    if (d_init)
      *d_init += T(0);
    return;
  }

  using ValueConst = typename ::std::iterator_traits<ValIt>::value_type;
  using Value = ::std::remove_const_t<ValueConst>;
  using Key = typename ::std::iterator_traits<KeyIt>::value_type;

  auto d_src_const_ptr = ::thrust::raw_pointer_cast((*d_values_first).base());
  auto d_src_ptr = const_cast<Value*>(d_src_const_ptr);
  ::thrust::device_ptr<Value> d_src_dev_ptr(d_src_ptr);

  auto d_dst_const_ptr = ::thrust::raw_pointer_cast((*d_result).base());
  auto d_dst_ptr = const_cast<Value*>(d_dst_const_ptr);
  ::thrust::device_ptr<Value> d_dst_dev_ptr(d_dst_ptr);

  if constexpr (::std::is_same_v<BinaryOp, ::thrust::plus<Value>> &&
                ::std::is_same_v<KeyEqual, ::thrust::equal_to<Key>>) {
    if (d_init) {
      *d_init +=
          ::thrust::reduce(d_dst_dev_ptr, d_dst_dev_ptr + n, Value(0), op);
    }

    ::thrust::device_vector<Value> suffix_sums(n);

    auto rev_keys_begin = ::thrust::make_reverse_iterator(keys_last);
    auto rev_keys_end = ::thrust::make_reverse_iterator(keys_first);
    auto rev_dy_begin = ::thrust::make_reverse_iterator(d_dst_dev_ptr + n);
    auto rev_dy_end = ::thrust::make_reverse_iterator(d_dst_dev_ptr);

    ::thrust::exclusive_scan_by_key(rev_keys_begin, rev_keys_end, rev_dy_begin,
                                    suffix_sums.begin(), Value(0),
                                    ::thrust::equal_to<Key>(), op);

    ::thrust::transform(d_src_dev_ptr, d_src_dev_ptr + n,
                        ::thrust::make_reverse_iterator(suffix_sums.end()),
                        d_src_dev_ptr, ::thrust::plus<Value>());

    ::thrust::fill(d_dst_dev_ptr, d_dst_dev_ptr + n, Value(0));
  } else {
    static_assert(::std::is_same_v<Value, void>,
                  "This predicate/operation is not supported by the custom "
                  "exclusive_scan_by_key_pullback.");
  }
}

template <typename KeyIt, typename ValIt, typename OutputIt, typename T>
clad::ValueAndAdjoint<OutputIt, OutputIt>
exclusive_scan_by_key_reverse_forw(KeyIt keys_first, KeyIt keys_last,
                                   ValIt values_first, OutputIt result, T init,
                                   KeyIt, KeyIt, ValIt, OutputIt, T) {
  using Key = typename ::std::iterator_traits<KeyIt>::value_type;
  using Value = typename ::std::iterator_traits<ValIt>::value_type;
  return {::thrust::exclusive_scan_by_key(
              keys_first, keys_last, values_first, result, init,
              ::thrust::equal_to<Key>(), ::thrust::plus<Value>()),
          {}};
}

template <typename KeyIt, typename ValIt, typename OutputIt, typename T,
          typename KeyEqual, typename BinaryOp>
clad::ValueAndAdjoint<OutputIt, OutputIt>
exclusive_scan_by_key_reverse_forw(KeyIt keys_first, KeyIt keys_last,
                                   ValIt values_first, OutputIt result, T init,
                                   KeyEqual pred, BinaryOp op, KeyIt, KeyIt,
                                   ValIt, OutputIt, T, KeyEqual, BinaryOp) {
  return {::thrust::exclusive_scan_by_key(keys_first, keys_last, values_first,
                                          result, init, pred, op),
          {}};
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
  } else if constexpr (detail::has_unary_operator_call_pullback<UnaryOp,
                                                                Value>::value) {
    ::thrust::device_vector<UnaryOp> d_op_storage;
    UnaryOp* d_op_device_ptr = nullptr;
    if (d_op) {
      d_op_storage.resize(1);
      d_op_storage[0] = *d_op;
      d_op_device_ptr = ::thrust::raw_pointer_cast(d_op_storage.data());
    }

    struct grad_functor {
      UnaryOp op;
      UnaryOp* d_op;
      CUDA_HOST_DEVICE void
      operator()(::thrust::tuple<Value&, Value&, const Value&> t) const {
        Value& d_x = ::thrust::get<0>(t);
        Value& d_y = ::thrust::get<1>(t);
        const Value& x = ::thrust::get<2>(t);
        Value d_x_local = Value(0);
        // Backprop through user functor
        op.operator_call_pullback(x, d_y, d_op, &d_x_local);
        d_x += d_x_local;
        d_y = 0;
      }
    };
    auto iter = ::thrust::make_zip_iterator(
        ::thrust::make_tuple(d_src_dev_ptr, d_dst_dev_ptr, first));
    ::thrust::for_each(iter, iter + n, grad_functor{op, d_op_device_ptr});
    if (d_op)
      *d_op = d_op_storage[0];
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
  } else if constexpr (detail::has_binary_operator_call_pullback<
                           BinaryOp, Value>::value) {
    ::thrust::device_vector<BinaryOp> d_op_storage;
    BinaryOp* d_op_device_ptr = nullptr;
    if (d_op) {
      d_op_storage.resize(1);
      d_op_storage[0] = *d_op;
      d_op_device_ptr = ::thrust::raw_pointer_cast(d_op_storage.data());
    }

    struct grad_functor {
      BinaryOp op;
      BinaryOp* d_op;
      CUDA_HOST_DEVICE void operator()(
          ::thrust::tuple<Value&, Value&, Value&, const Value&, const Value&> t)
          const {
        Value& d_x1 = ::thrust::get<0>(t);
        Value& d_x2 = ::thrust::get<1>(t);
        Value& d_y = ::thrust::get<2>(t);
        const Value& x1 = ::thrust::get<3>(t);
        const Value& x2 = ::thrust::get<4>(t);
        Value d_x1_local = Value(0);
        Value d_x2_local = Value(0);
        // Backprop through user functor
        op.operator_call_pullback(x1, x2, d_y, d_op, &d_x1_local, &d_x2_local);
        d_x1 += d_x1_local;
        d_x2 += d_x2_local;
        d_y = 0;
      }
    };
    auto iter = ::thrust::make_zip_iterator(::thrust::make_tuple(
        d_first1_dev_ptr, d_first2_dev_ptr, d_result_dev_ptr, first1, first2));
    ::thrust::for_each(iter, iter + n, grad_functor{op, d_op_device_ptr});
    if (d_op)
      *d_op = d_op_storage[0];
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

template <typename KeyInputIt, typename ValueInputIt, typename KeyOutputIt,
          typename ValueOutputIt>
clad::ValueAndAdjoint<::thrust::pair<KeyOutputIt, ValueOutputIt>,
                      ::thrust::pair<KeyOutputIt, ValueOutputIt>>
reduce_by_key_reverse_forw(KeyInputIt keys_first, KeyInputIt keys_last,
                           ValueInputIt values_first, KeyOutputIt keys_output,
                           ValueOutputIt values_output, KeyInputIt, KeyInputIt,
                           ValueInputIt, KeyOutputIt, ValueOutputIt) {
  using Key = typename ::std::iterator_traits<KeyInputIt>::value_type;
  using Value = typename ::std::iterator_traits<ValueInputIt>::value_type;
  auto p = ::thrust::reduce_by_key(
      keys_first, keys_last, values_first, keys_output, values_output,
      ::thrust::equal_to<Key>(), ::thrust::plus<Value>());
  return {p, {}};
}

template <typename KeyInputIt, typename ValueInputIt, typename KeyOutputIt,
          typename ValueOutputIt, typename BinaryPredicate, typename BinaryOp>
clad::ValueAndAdjoint<::thrust::pair<KeyOutputIt, ValueOutputIt>,
                      ::thrust::pair<KeyOutputIt, ValueOutputIt>>
reduce_by_key_reverse_forw(KeyInputIt keys_first, KeyInputIt keys_last,
                           ValueInputIt values_first, KeyOutputIt keys_output,
                           ValueOutputIt values_output, BinaryPredicate pred,
                           BinaryOp op, KeyInputIt, KeyInputIt, ValueInputIt,
                           KeyOutputIt, ValueOutputIt, BinaryPredicate,
                           BinaryOp) {
  auto p = ::thrust::reduce_by_key(keys_first, keys_last, values_first,
                                   keys_output, values_output, pred, op);
  return {p, {}};
}

template <typename KeyInputIt, typename ValueInputIt, typename KeyOutputIt,
          typename ValueOutputIt, typename BinaryPredicate, typename BinaryOp>
void reduce_by_key_pullback(
    KeyInputIt keys_first, KeyInputIt keys_last, ValueInputIt /*values_first*/,
    KeyOutputIt /*keys_output*/, ValueOutputIt values_output,
    BinaryPredicate pred, BinaryOp /*op*/,
    ::thrust::pair<KeyOutputIt, ValueOutputIt> /*d_return*/,
    KeyInputIt* /*d_kf*/, KeyInputIt* /*d_kl*/, ValueInputIt* d_values_first,
    KeyOutputIt* /*d_keys_output*/, ValueOutputIt* d_values_output,
    BinaryPredicate* /*d_pred*/, BinaryOp* /*d_op*/) {
  using ValueConst = typename ::std::iterator_traits<ValueInputIt>::value_type;
  using Value = ::std::remove_const_t<ValueConst>;
  using KeyConst = typename ::std::iterator_traits<KeyInputIt>::value_type;
  using Key = ::std::remove_const_t<KeyConst>;

  static_assert(
      ::std::is_same_v<BinaryPredicate, ::thrust::equal_to<Key>>,
      "Only thrust::equal_to<Key> is supported for reduce_by_key pullback");
  static_assert(
      ::std::is_same_v<BinaryOp, ::thrust::plus<Value>> ||
          ::std::is_same_v<BinaryOp, void>,
      "Only thrust::plus<Value> is supported for reduce_by_key pullback");

  const ::std::size_t n = ::thrust::distance(keys_first, keys_last);
  if (n == 0)
    return;

  // Access adjoint buffers
  auto d_in_const = ::thrust::raw_pointer_cast((*d_values_first).base());
  auto d_in_ptr = const_cast<Value*>(d_in_const);
  ::thrust::device_ptr<Value> d_in(d_in_ptr);

  auto d_out_const = ::thrust::raw_pointer_cast((*d_values_output).base());
  auto d_out_ptr = const_cast<Value*>(d_out_const);
  ::thrust::device_ptr<Value> d_out(d_out_ptr);

  // Build group start flags: 1 if i==0 or keys[i] not equal keys[i-1]
  using Index = ::std::size_t;
  auto kptr_const = ::thrust::raw_pointer_cast(keys_first.base());
  auto kptr = const_cast<Key*>(kptr_const);
  ::thrust::device_ptr<Key> keys(kptr);

  ::thrust::device_vector<int> flags(n);
  struct group_start_functor {
    Key* keys{};
    BinaryPredicate pred{};
    CUDA_HOST_DEVICE int operator()(Index i) const {
      if (i == 0)
        return 1;
      return pred(keys[i], keys[i - 1]) ? 0 : 1;
    }
  };
  ::thrust::transform(::thrust::counting_iterator<Index>(0),
                      ::thrust::counting_iterator<Index>(n), flags.begin(),
                      group_start_functor{keys.get(), pred});

  // Inclusive scan to get group ids in 1..m, then subtract 1 to [0..m-1]
  ::thrust::device_vector<int> group_id(n);
  ::thrust::inclusive_scan(flags.begin(), flags.end(), group_id.begin());
  ::thrust::transform(group_id.begin(), group_id.end(), group_id.begin(),
                      [] __device__(int x) { return x - 1; });

  // Scatter output adjoints back to inputs: d_in[i] += d_out[group_id[i]]
  struct scatter_functor {
    Value* d_in{};
    const Value* d_out{};
    const int* gid{};
    CUDA_HOST_DEVICE void operator()(Index i) const {
      d_in[i] += d_out[gid[i]];
    }
  };
  auto gid_ptr = ::thrust::raw_pointer_cast(group_id.data());
  ::thrust::for_each(::thrust::counting_iterator<Index>(0),
                     ::thrust::counting_iterator<Index>(n),
                     scatter_functor{d_in.get(), d_out.get(), gid_ptr});

  // Clear output adjoints
  // m = sum(flags)
  int m = ::thrust::reduce(flags.begin(), flags.end(), 0);
  if (m > 0)
    ::thrust::fill(d_out, d_out + m, Value(0));
}

// pullback (default predicate/op = equal_to/plus)
template <typename KeyInputIt, typename ValueInputIt, typename KeyOutputIt,
          typename ValueOutputIt>
void reduce_by_key_pullback(
    KeyInputIt keys_first, KeyInputIt keys_last, ValueInputIt values_first,
    KeyOutputIt keys_output, ValueOutputIt values_output,
    ::thrust::pair<KeyOutputIt, ValueOutputIt> /*d_return*/, KeyInputIt* d_kf,
    KeyInputIt* d_kl, ValueInputIt* d_values_first, KeyOutputIt* d_keys_output,
    ValueOutputIt* d_values_output) {
  using ValueConst = typename ::std::iterator_traits<ValueInputIt>::value_type;
  using Value = ::std::remove_const_t<ValueConst>;
  using KeyConst = typename ::std::iterator_traits<KeyInputIt>::value_type;
  using Key = ::std::remove_const_t<KeyConst>;

  ::thrust::equal_to<Key> pred;
  ::thrust::equal_to<Key>* d_pred = nullptr;
  ::thrust::plus<Value> op;
  ::thrust::plus<Value>* d_op = nullptr;
  reduce_by_key_pullback(keys_first, keys_last, values_first, keys_output,
                         values_output, pred, op, /*d_return*/ {}, d_kf, d_kl,
                         d_values_first, d_keys_output, d_values_output, d_pred,
                         d_op);
}

template <typename InputIt, typename OutputIt>
void adjacent_difference_pullback(InputIt first, InputIt last, OutputIt result,
                                  OutputIt /*d_return*/, InputIt* d_first,
                                  InputIt* d_last, OutputIt* d_result) {
  using Value = typename ::std::iterator_traits<InputIt>::value_type;
  ::thrust::minus<Value> op;
  ::thrust::minus<Value>* d_op = nullptr;
  adjacent_difference_pullback(first, last, result, op, {}, d_first, d_last,
                               d_result, d_op);
}

template <typename InputIt, typename OutputIt, typename BinaryOp>
void adjacent_difference_pullback(InputIt first, InputIt last, OutputIt result,
                                  BinaryOp /*op*/, OutputIt /*d_return*/,
                                  InputIt* d_first, InputIt* d_last,
                                  OutputIt* d_result, BinaryOp* /*d_op*/) {
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

  // First, accumulate direct contribution: d_x[i] += d_y[i]
  ::thrust::transform(d_src_dev_ptr, d_src_dev_ptr + n, d_dst_dev_ptr,
                      d_src_dev_ptr, ::thrust::plus<Value>());

  if constexpr (::std::is_same_v<BinaryOp, ::thrust::minus<Value>> ||
                ::std::is_same_v<BinaryOp, ::thrust::plus<Value>>) {
    if (n > 1) {
      using ShiftOp = ::std::conditional_t<
          ::std::is_same_v<BinaryOp, ::thrust::minus<Value>>,
          ::thrust::minus<Value>, ::thrust::plus<Value>>;
      ::thrust::transform(d_src_dev_ptr, d_src_dev_ptr + (n - 1),
                          d_dst_dev_ptr + 1, d_src_dev_ptr, ShiftOp());
    }
  } else {
    static_assert(::std::is_same_v<Value, void>,
                  "This binary operation is not supported by the custom "
                  "adjacent_difference_pullback.");
  }

  // Clear output adjoints
  ::thrust::fill(d_dst_dev_ptr, d_dst_dev_ptr + n, Value(0));
}

template <typename InputIt, typename OutputIt>
clad::ValueAndAdjoint<OutputIt, OutputIt>
adjacent_difference_reverse_forw(InputIt first, InputIt last, OutputIt result,
                                 InputIt, InputIt, OutputIt) {
  using Value = typename ::std::iterator_traits<InputIt>::value_type;
  return {::thrust::adjacent_difference(first, last, result,
                                        ::thrust::minus<Value>()),
          {}};
}

template <typename InputIt, typename OutputIt, typename BinaryOp>
clad::ValueAndAdjoint<OutputIt, OutputIt>
adjacent_difference_reverse_forw(InputIt first, InputIt last, OutputIt result,
                                 BinaryOp op, InputIt, InputIt, OutputIt,
                                 BinaryOp) {
  return {::thrust::adjacent_difference(first, last, result, op), {}};
}

namespace detail {
inline ::std::vector<::thrust::device_vector<::std::size_t>>&
permutation_stack() {
  static ::std::vector<::thrust::device_vector<::std::size_t>> stack;
  return stack;
}
} // namespace detail

template <typename KeyIterator, typename ValueIterator>
void sort_by_key_reverse_forw(KeyIterator keys_first, KeyIterator keys_last,
                              ValueIterator values_first, KeyIterator,
                              KeyIterator, ValueIterator) {
  const ::std::size_t n = ::thrust::distance(keys_first, keys_last);
  if (n > 0) {
    ::thrust::device_vector<::std::size_t> perm(n);
    ::thrust::sequence(perm.begin(), perm.end());
    struct index_comp_less {
      KeyIterator keys_begin;
      CUDA_HOST_DEVICE bool operator()(::std::size_t a, ::std::size_t b) const {
        return keys_begin[a] < keys_begin[b];
      }
    };
    ::thrust::sort(perm.begin(), perm.end(), index_comp_less{keys_first});
    detail::permutation_stack().push_back(::std::move(perm));
  }
  ::thrust::sort_by_key(keys_first, keys_last, values_first);
}

template <typename KeyIterator, typename ValueIterator, typename Compare>
void sort_by_key_reverse_forw(KeyIterator keys_first, KeyIterator keys_last,
                              ValueIterator values_first, Compare comp,
                              KeyIterator, KeyIterator, ValueIterator,
                              Compare) {
  const ::std::size_t n = ::thrust::distance(keys_first, keys_last);
  if (n > 0) {
    ::thrust::device_vector<::std::size_t> perm(n);
    ::thrust::sequence(perm.begin(), perm.end());
    struct index_comp_with {
      KeyIterator keys_begin;
      Compare comp;
      CUDA_HOST_DEVICE bool operator()(::std::size_t a, ::std::size_t b) const {
        return comp(keys_begin[a], keys_begin[b]);
      }
    };
    ::thrust::sort(perm.begin(), perm.end(), index_comp_with{keys_first, comp});
    detail::permutation_stack().push_back(::std::move(perm));
  }
  ::thrust::sort_by_key(keys_first, keys_last, values_first, comp);
}

// pullback without comparator
template <typename KeyIterator, typename ValueIterator>
void sort_by_key_pullback(KeyIterator keys_first, KeyIterator keys_last,
                          ValueIterator values_first, KeyIterator* d_keys_first,
                          KeyIterator* d_keys_last,
                          ValueIterator* d_values_first) {
  using ValueConst = typename ::std::iterator_traits<ValueIterator>::value_type;
  using Value = ::std::remove_const_t<ValueConst>;
  using KeyConst = typename ::std::iterator_traits<KeyIterator>::value_type;
  using Key = ::std::remove_const_t<KeyConst>;

  const ::std::size_t n = ::thrust::distance(keys_first, keys_last);
  if (n == 0)
    return;

  // Retrieve permutation mapping sorted position j -> original index i
  auto& stack = detail::permutation_stack();
  ::thrust::device_vector<::std::size_t> perm = ::std::move(stack.back());
  stack.pop_back();

  // Build device pointers to adjoint buffers
  auto d_vals_const_ptr = ::thrust::raw_pointer_cast((*d_values_first).base());
  auto d_vals_ptr = const_cast<Value*>(d_vals_const_ptr);
  ::thrust::device_ptr<Value> d_vals(d_vals_ptr);

  // Make a copy of current (sorted order) adjoints, then scatter-add into
  // original positions and clear the sorted adjoints.
  ::thrust::device_vector<Value> dvals_tmp(n);
  ::thrust::copy(d_vals, d_vals + n, dvals_tmp.begin());
  ::thrust::fill(d_vals, d_vals + n, Value(0));

  auto out_perm_begin =
      ::thrust::make_permutation_iterator(d_vals, perm.begin());
  auto out_perm_end =
      ::thrust::make_permutation_iterator(d_vals, perm.begin() + n);
  ::thrust::transform(out_perm_begin, out_perm_end, dvals_tmp.begin(),
                      out_perm_begin, ::thrust::plus<Value>());

  // Keys do not receive gradients.
  if (d_keys_first && d_keys_last) {
    auto d_keys_const_ptr = ::thrust::raw_pointer_cast((*d_keys_first).base());
    auto d_keys_ptr = const_cast<Key*>(d_keys_const_ptr);
    ::thrust::device_ptr<Key> d_keys(d_keys_ptr);
    ::thrust::fill(d_keys, d_keys + n, Key(0));
  }
}

// pullback with comparator
template <typename KeyIterator, typename ValueIterator, typename Compare>
void sort_by_key_pullback(KeyIterator keys_first, KeyIterator keys_last,
                          ValueIterator values_first, Compare comp,
                          KeyIterator* d_keys_first, KeyIterator* d_keys_last,
                          ValueIterator* d_values_first, Compare* /*d_comp*/) {
  using ValueConst = typename ::std::iterator_traits<ValueIterator>::value_type;
  using Value = ::std::remove_const_t<ValueConst>;
  using KeyConst = typename ::std::iterator_traits<KeyIterator>::value_type;
  using Key = ::std::remove_const_t<KeyConst>;

  const ::std::size_t n = ::thrust::distance(keys_first, keys_last);
  if (n == 0)
    return;

  auto& stack = detail::permutation_stack();
  ::thrust::device_vector<::std::size_t> perm = ::std::move(stack.back());
  stack.pop_back();

  auto d_vals_const_ptr = ::thrust::raw_pointer_cast((*d_values_first).base());
  auto d_vals_ptr = const_cast<Value*>(d_vals_const_ptr);
  ::thrust::device_ptr<Value> d_vals(d_vals_ptr);

  ::thrust::device_vector<Value> dvals_tmp(n);
  ::thrust::copy(d_vals, d_vals + n, dvals_tmp.begin());
  ::thrust::fill(d_vals, d_vals + n, Value(0));

  auto out_perm_begin =
      ::thrust::make_permutation_iterator(d_vals, perm.begin());
  auto out_perm_end =
      ::thrust::make_permutation_iterator(d_vals, perm.begin() + n);
  ::thrust::transform(out_perm_begin, out_perm_end, dvals_tmp.begin(),
                      out_perm_begin, ::thrust::plus<Value>());

  if (d_keys_first && d_keys_last) {
    auto d_keys_const_ptr = ::thrust::raw_pointer_cast((*d_keys_first).base());
    auto d_keys_ptr = const_cast<Key*>(d_keys_const_ptr);
    ::thrust::device_ptr<Key> d_keys(d_keys_ptr);
    ::thrust::fill(d_keys, d_keys + n, Key(0));
  }
}

} // namespace clad::custom_derivatives::thrust

#endif // CLAD_DIFFERENTIATOR_THRUSTDERIVATIVES_H
