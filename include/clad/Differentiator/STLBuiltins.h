#ifndef CLAD_STL_BUILTINS_H
#define CLAD_STL_BUILTINS_H

#include <clad/Differentiator/BuiltinDerivatives.h>
#include <initializer_list>
#include <vector>

namespace clad {
namespace custom_derivatives {
namespace class_functions {

template <typename T>
void clear_pushforward(::std::vector<T>* v, ::std::vector<T>* d_v) {
  d_v->clear();
  v->clear();
}

template <typename T>
void resize_pushforward(::std::vector<T>* v, unsigned sz, ::std::vector<T>* d_v,
                        unsigned d_sz) {
  d_v->resize(sz, T());
  v->resize(sz);
}

template <typename T, typename U>
void resize_pushforward(::std::vector<T>* v, unsigned sz, U val,
                        ::std::vector<T>* d_v, unsigned d_sz, U d_val) {
  d_v->resize(sz, d_val);
  v->resize(sz, val);
}

template <typename T>
ValueAndPushforward<typename ::std::initializer_list<T>::iterator,
                    typename ::std::initializer_list<T>::iterator>
begin_pushforward(::std::initializer_list<T>* il,
                  ::std::initializer_list<T>* d_il) {
  return {il->begin(), d_il->begin()};
}

template <typename T>
ValueAndPushforward<typename ::std::initializer_list<T>::iterator,
                    typename ::std::initializer_list<T>::iterator>
end_pushforward(const ::std::initializer_list<T>* il,
                const ::std::initializer_list<T>* d_il) {
  return {il->end(), d_il->end()};
}

template <typename T>
clad::ValueAndPushforward<::std::vector<T>, ::std::vector<T>>
constructor_pushforward(
    ConstructorPushforwardTag<::std::vector<T>>,
    typename ::std::vector<T>::size_type n,
    const typename ::std::vector<T>::allocator_type& alloc,
    typename ::std::vector<T>::size_type d_n,
    const typename ::std::vector<T>::allocator_type& d_alloc) {
  ::std::vector<T> v(n, alloc);
  ::std::vector<T> d_v(n, 0, alloc);
  return {v, d_v};
}

template <typename T>
clad::ValueAndPushforward<::std::vector<T>, ::std::vector<T>>
constructor_pushforward(
    ConstructorPushforwardTag<::std::vector<T>>,
    typename ::std::vector<T>::size_type n, T val,
    const typename ::std::vector<T>::allocator_type& alloc,
    typename ::std::vector<T>::size_type d_n, T d_val,
    const typename ::std::vector<T>::allocator_type& d_alloc) {
  ::std::vector<T> v(n, val, alloc);
  ::std::vector<T> d_v(n, d_val, alloc);
  return {v, d_v};
}

template <typename T>
clad::ValueAndPushforward<::std::vector<T>, ::std::vector<T>>
constructor_pushforward(
    ConstructorPushforwardTag<::std::vector<T>>,
    ::std::initializer_list<T> list,
    const typename ::std::vector<T>::allocator_type& alloc,
    ::std::initializer_list<T> dlist,
    const typename ::std::vector<T>::allocator_type& dalloc) {
  ::std::vector<T> v(list, alloc);
  ::std::vector<T> d_v(dlist, dalloc);
  return {v, d_v};
}

template <typename T>
clad::ValueAndPushforward<::std::vector<T>, ::std::vector<T>>
constructor_pushforward(ConstructorPushforwardTag<::std::vector<T>>,
                        typename ::std::vector<T>::size_type n,
                        typename ::std::vector<T>::size_type d_n) {
  ::std::vector<T> v(n);
  ::std::vector<T> d_v(n, 0);
  return {v, d_v};
}

template <typename T>
clad::ValueAndPushforward<::std::vector<T>, ::std::vector<T>>
constructor_pushforward(ConstructorPushforwardTag<::std::vector<T>>,
                        typename ::std::vector<T>::size_type n, T val,
                        typename ::std::vector<T>::size_type d_n, T d_val) {
  ::std::vector<T> v(n, val);
  ::std::vector<T> d_v(n, d_val);
  return {v, d_v};
}

template <typename T>
clad::ValueAndPushforward<::std::vector<T>, ::std::vector<T>>
constructor_pushforward(ConstructorPushforwardTag<::std::vector<T>>,
                        ::std::initializer_list<T> list,
                        ::std::initializer_list<T> dlist) {
  ::std::vector<T> v(list);
  ::std::vector<T> d_v(dlist);
  return {v, d_v};
}

template <typename T>
ValueAndPushforward<T&, T&>
operator_subscript_pushforward(::std::vector<T>* v, unsigned idx,
                               ::std::vector<T>* d_v, unsigned d_idx) {
  return {(*v)[idx], (*d_v)[idx]};
}

template <typename T>
ValueAndPushforward<const T&, const T&>
operator_subscript_pushforward(const ::std::vector<T>* v, unsigned idx,
                               const ::std::vector<T>* d_v, unsigned d_idx) {
  return {(*v)[idx], (*d_v)[idx]};
}

template <typename T, ::std::size_t N>
constexpr clad::ValueAndPushforward<T&, T&>
operator_subscript_pushforward(::std::array<T, N>* a, ::std::size_t i,
                               ::std::array<T, N>* d_a,
                               ::std::size_t d_i) noexcept {
  return {(*a)[i], (*d_a)[i]};
}

template <typename T, ::std::size_t N>
constexpr clad::ValueAndPushforward<T&, T&>
at_pushforward(::std::array<T, N>* a, ::std::size_t i, ::std::array<T, N>* d_a,
               ::std::size_t d_i) noexcept {
  return {(*a)[i], (*d_a)[i]};
}

template <typename T, ::std::size_t N>
constexpr clad::ValueAndPushforward<const T&, const T&>
operator_subscript_pushforward(const ::std::array<T, N>* a, ::std::size_t i,
                               const ::std::array<T, N>* d_a,
                               ::std::size_t d_i) noexcept {
  return {(*a)[i], (*d_a)[i]};
}

template <typename T, ::std::size_t N>
constexpr clad::ValueAndPushforward<const T&, const T&>
at_pushforward(const ::std::array<T, N>* a, ::std::size_t i,
               const ::std::array<T, N>* d_a, ::std::size_t d_i) noexcept {
  return {(*a)[i], (*d_a)[i]};
}

template <typename T, ::std::size_t N>
constexpr clad::ValueAndPushforward<::std::array<T, N>&, ::std::array<T, N>&>
operator_equal_pushforward(::std::array<T, N>* a,
                           const ::std::array<T, N>& param,
                           ::std::array<T, N>* d_a,
                           const ::std::array<T, N>& d_param) noexcept {
  (*a) = param;
  (*d_a) = d_param;
  return {*a, *d_a};
}

template <typename T, ::std::size_t N>
inline constexpr clad::ValueAndPushforward<const T&, const T&>
front_pushforward(const ::std::array<T, N>* a,
                  const ::std::array<T, N>* d_a) noexcept {
  return {a->front(), d_a->front()};
}

template <typename T, ::std::size_t N>
inline constexpr clad::ValueAndPushforward<T&, T&>
front_pushforward(::std::array<T, N>* a, ::std::array<T, N>* d_a) noexcept {
  return {a->front(), d_a->front()};
}

template <typename T, ::std::size_t N>
inline constexpr clad::ValueAndPushforward<const T&, const T&>
back_pushforward(const ::std::array<T, N>* a,
                 const ::std::array<T, N>* d_a) noexcept {
  return {a->back(), d_a->back()};
}

template <typename T, ::std::size_t N>
inline constexpr clad::ValueAndPushforward<T&, T&>
back_pushforward(::std::array<T, N>* a, ::std::array<T, N>* d_a) noexcept {
  return {a->back(), d_a->back()};
}

template <typename T, ::std::size_t N>
void fill_pushforward(::std::array<T, N>* a, const T& u,
                      ::std::array<T, N>* d_a, const T& d_u) {
  a->fill(u);
  d_a->fill(d_u);
}

template <typename T, typename U>
void push_back_reverse_forw(::std::vector<T>* v, U val, ::std::vector<T>* d_v,
                            U d_val) {
  v->push_back(val);
  d_v->push_back(0);
}

template <typename T, typename U>
void push_back_pullback(::std::vector<T>* v, U val, ::std::vector<T>* d_v,
                        U* d_val) {
  *d_val += d_v->back();
  d_v->pop_back();
}

template <typename T>
clad::ValueAndAdjoint<T&, T&> operator_subscript_reverse_forw(
    ::std::vector<T>* vec, typename ::std::vector<T>::size_type idx,
    ::std::vector<T>* d_vec, typename ::std::vector<T>::size_type d_idx) {
  return {(*vec)[idx], (*d_vec)[idx]};
}

template <typename T, typename P>
void operator_subscript_pullback(::std::vector<T>* vec,
                                 typename ::std::vector<T>::size_type idx,
                                 P d_y, ::std::vector<T>* d_vec,
                                 typename ::std::vector<T>::size_type* d_idx) {
  (*d_vec)[idx] += d_y;
}

template <typename T, typename S, typename U>
::clad::ValueAndAdjoint<::std::vector<T>, ::std::vector<T>>
constructor_reverse_forw(::clad::ConstructorReverseForwTag<::std::vector<T>>,
                         S count, U val,
                         typename ::std::vector<T>::allocator_type alloc,
                         S d_count, U d_val,
                         typename ::std::vector<T>::allocator_type d_alloc) {
  ::std::vector<T> v(count, val);
  ::std::vector<T> d_v(count, 0);
  return {v, d_v};
}

template <typename T, typename S, typename U>
void constructor_pullback(::std::vector<T>* v, S count, U val,
                          typename ::std::vector<T>::allocator_type alloc,
                          ::std::vector<T>* d_v, S* d_count, U* d_val,
                          typename ::std::vector<T>::allocator_type* d_alloc) {
  for (unsigned i = 0; i < count; ++i)
    *d_val += (*d_v)[i];
  d_v->clear();
}

template <typename T, ::std::size_t N>
clad::ValueAndAdjoint<T&, T&> operator_subscript_reverse_forw(
    ::std::array<T, N>* arr, typename ::std::array<T, N>::size_type idx,
    ::std::array<T, N>* d_arr, typename ::std::array<T, N>::size_type d_idx) {
  return {(*arr)[idx], (*d_arr)[idx]};
}
template <typename T, ::std::size_t N, typename P>
void operator_subscript_pullback(
    ::std::array<T, N>* arr, typename ::std::array<T, N>::size_type idx, P d_y,
    ::std::array<T, N>* d_arr, typename ::std::array<T, N>::size_type* d_idx) {
  (*d_arr)[idx] += d_y;
}
template <typename T, ::std::size_t N>
clad::ValueAndAdjoint<T&, T&> at_reverse_forw(
    ::std::array<T, N>* arr, typename ::std::array<T, N>::size_type idx,
    ::std::array<T, N>* d_arr, typename ::std::array<T, N>::size_type d_idx) {
  return {(*arr)[idx], (*d_arr)[idx]};
}
template <typename T, ::std::size_t N, typename P>
void at_pullback(::std::array<T, N>* arr,
                 typename ::std::array<T, N>::size_type idx, P d_y,
                 ::std::array<T, N>* d_arr,
                 typename ::std::array<T, N>::size_type* d_idx) {
  (*d_arr)[idx] += d_y;
}
template <typename T, ::std::size_t N>
void fill_reverse_forw(::std::array<T, N>* a,
                       const typename ::std::array<T, N>::value_type& u,
                       ::std::array<T, N>* d_a,
                       const typename ::std::array<T, N>::value_type& d_u) {
  a->fill(u);
  d_a->fill(0);
}
template <typename T, ::std::size_t N>
void fill_pullback(::std::array<T, N>* arr,
                   const typename ::std::array<T, N>::value_type& u,
                   ::std::array<T, N>* d_arr,
                   typename ::std::array<T, N>::value_type* d_u) {
  for (size_t i = 0; i < N; ++i) {
    typename ::std::array<T, N>::value_type r_d0 = (*d_arr)[i];
    (*d_arr)[i] = 0;
    *d_u += r_d0;
  }
}
template <typename T, ::std::size_t N>
clad::ValueAndAdjoint<T&, T&>
back_reverse_forw(::std::array<T, N>* arr, ::std::array<T, N>* d_arr) noexcept {
  return {arr->back(), d_arr->back()};
}
template <typename T, ::std::size_t N>
void back_pullback(::std::array<T, N>* arr,
                   typename ::std::array<T, N>::value_type d_u,
                   ::std::array<T, N>* d_arr) noexcept {
  (*d_arr)[d_arr->size() - 1] += d_u;
}
template <typename T, ::std::size_t N>
clad::ValueAndAdjoint<T&, T&>
front_reverse_forw(::std::array<T, N>* arr,
                   ::std::array<T, N>* d_arr) noexcept {
  return {arr->front(), d_arr->front()};
}
template <typename T, ::std::size_t N>
void front_pullback(::std::array<T, N>* arr,
                    typename ::std::array<T, N>::value_type d_u,
                    ::std::array<T, N>* d_arr) {
  (*d_arr)[0] += d_u;
}
template <typename T, ::std::size_t N>
void size_pullback(::std::array<T, N>* a, ::std::array<T, N>* d_a) noexcept {}
template <typename T, ::std::size_t N>
::clad::ValueAndAdjoint<::std::array<T, N>, ::std::array<T, N>>
constructor_reverse_forw(::clad::ConstructorReverseForwTag<::std::array<T, N>>,
                         const ::std::array<T, N>& arr,
                         const ::std::array<T, N>& d_arr) {
  ::std::array<T, N> a = arr;
  ::std::array<T, N> d_a = d_arr;
  return {a, d_a};
}
template <typename T, ::std::size_t N>
void constructor_pullback(::std::array<T, N>* a, const ::std::array<T, N>& arr,
                          ::std::array<T, N>* d_a, ::std::array<T, N>* d_arr) {
  for (size_t i = 0; i < N; ++i)
    (*d_arr)[i] += (*d_a)[i];
}

} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad

#endif // CLAD_STL_BUILTINS_H
