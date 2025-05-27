#ifndef CLAD_STL_BUILTINS_H
#define CLAD_STL_BUILTINS_H

#include <array>
#include <clad/Differentiator/Array.h>
#include <clad/Differentiator/BuiltinDerivatives.h>
#include <clad/Differentiator/FunctionTraits.h>
#include <initializer_list>
#include <memory>
#include <tuple>
#include <vector>

namespace clad {

// zero_init specializations

// std::pair<T1, T2> is almost trivially copyable. Specialize it.
template <class T1, class T2> void zero_init(std::pair<T1, T2>& p) {
  zero_init(p.first);
  zero_init(p.second);
}

//  is almost trivially copyable. Specialize it.
template <class T> void zero_init(typename std::allocator<T>&) {
  // do nothing, unclear if allocators have differentiable properties.
}

namespace custom_derivatives {
namespace class_functions {

// vector forward mode

template <typename T>
void clear_pushforward(::std::vector<T>* v, ::std::vector<T>* d_v) {
  d_v->clear();
  v->clear();
}

template <typename T>
void resize_pushforward(::std::vector<T>* v,
                        typename ::std::vector<T>::size_type sz,
                        ::std::vector<T>* d_v,
                        typename ::std::vector<T>::size_type d_sz) {
  d_v->resize(sz, T());
  v->resize(sz);
}

template <typename T, typename U>
void resize_pushforward(::std::vector<T>* v,
                        typename ::std::vector<T>::size_type sz, U val,
                        ::std::vector<T>* d_v,
                        typename ::std::vector<T>::size_type d_sz, U d_val) {
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
ValueAndPushforward<T&, T&> operator_subscript_pushforward(
    ::std::vector<T>* v, typename ::std::vector<T>::size_type idx,
    ::std::vector<T>* d_v, typename ::std::vector<T>::size_type d_idx) {
  return {(*v)[idx], (*d_v)[idx]};
}

template <typename T>
ValueAndPushforward<const T&, const T&> operator_subscript_pushforward(
    const ::std::vector<T>* v, typename ::std::vector<T>::size_type idx,
    const ::std::vector<T>* d_v, typename ::std::vector<T>::size_type d_idx) {
  return {(*v)[idx], (*d_v)[idx]};
}

template <typename T>
ValueAndPushforward<T&, T&>
at_pushforward(::std::vector<T>* v, typename ::std::vector<T>::size_type idx,
               ::std::vector<T>* d_v,
               typename ::std::vector<T>::size_type d_idx) {
  return {(*v)[idx], (*d_v)[idx]};
}

template <typename T>
ValueAndPushforward<const T&, const T&> at_pushforward(
    const ::std::vector<T>* v, typename ::std::vector<T>::size_type idx,
    const ::std::vector<T>* d_v, typename ::std::vector<T>::size_type d_idx) {
  return {(*v)[idx], (*d_v)[idx]};
}

template <typename T>
clad::ValueAndPushforward<::std::vector<T>&, ::std::vector<T>&>
operator_equal_pushforward(::std::vector<T>* a, const ::std::vector<T>& param,
                           ::std::vector<T>* d_a,
                           const ::std::vector<T>& d_param) noexcept {
  (*a) = param;
  (*d_a) = d_param;
  return {*a, *d_a};
}

template <typename T>
inline clad::ValueAndPushforward<const T&, const T&>
front_pushforward(const ::std::vector<T>* a,
                  const ::std::vector<T>* d_a) noexcept {
  return {a->front(), d_a->front()};
}

template <typename T>
inline clad::ValueAndPushforward<T&, T&>
front_pushforward(::std::vector<T>* a, ::std::vector<T>* d_a) noexcept {
  return {a->front(), d_a->front()};
}

template <typename T>
inline clad::ValueAndPushforward<const T&, const T&>
back_pushforward(const ::std::vector<T>* a,
                 const ::std::vector<T>* d_a) noexcept {
  return {a->back(), d_a->back()};
}

template <typename T>
inline clad::ValueAndPushforward<T&, T&>
back_pushforward(::std::vector<T>* a, ::std::vector<T>* d_a) noexcept {
  return {a->back(), d_a->back()};
}

template <typename T>
ValueAndPushforward<typename ::std::vector<T>::iterator,
                    typename ::std::vector<T>::iterator>
begin_pushforward(::std::vector<T>* v, ::std::vector<T>* d_v) {
  return {v->begin(), d_v->begin()};
}

template <typename T>
ValueAndPushforward<typename ::std::vector<T>::iterator,
                    typename ::std::vector<T>::iterator>
end_pushforward(::std::vector<T>* v, ::std::vector<T>* d_v) {
  return {v->end(), d_v->end()};
}

template <typename T>
ValueAndPushforward<typename ::std::vector<T>::iterator,
                    typename ::std::vector<T>::iterator>
erase_pushforward(::std::vector<T>* v,
                  typename ::std::vector<T>::const_iterator pos,
                  ::std::vector<T>* d_v,
                  typename ::std::vector<T>::const_iterator d_pos) {
  return {v->erase(pos), d_v->erase(d_pos)};
}

template <typename T, typename U>
ValueAndPushforward<typename ::std::vector<T>::iterator,
                    typename ::std::vector<T>::iterator>
insert_pushforward(::std::vector<T>* v,
                   typename ::std::vector<T>::const_iterator pos, U u,
                   ::std::vector<T>* d_v,
                   typename ::std::vector<T>::const_iterator d_pos, U d_u) {
  return {v->insert(pos, u), d_v->insert(d_pos, d_u)};
}

template <typename T, typename U>
ValueAndPushforward<typename ::std::vector<T>::iterator,
                    typename ::std::vector<T>::iterator>
insert_pushforward(::std::vector<T>* v,
                   typename ::std::vector<T>::const_iterator pos,
                   ::std::initializer_list<U> list, ::std::vector<T>* d_v,
                   typename ::std::vector<T>::const_iterator d_pos,
                   ::std::initializer_list<U> d_list) {
  return {v->insert(pos, list), d_v->insert(d_pos, d_list)};
}

template <typename T, typename U>
ValueAndPushforward<typename ::std::vector<T>::iterator,
                    typename ::std::vector<T>::iterator>
insert_pushforward(::std::vector<T>* v,
                   typename ::std::vector<T>::const_iterator pos, U first,
                   U last, ::std::vector<T>* d_v,
                   typename ::std::vector<T>::const_iterator d_pos, U d_first,
                   U d_last) {
  return {v->insert(pos, first, last), d_v->insert(d_pos, d_first, d_last)};
}

template <typename T, typename U>
void assign_pushforward(::std::vector<T>* v,
                        typename ::std::vector<T>::size_type n, const U& val,
                        ::std::vector<T>* d_v,
                        typename ::std::vector<T>::size_type /*d_n*/,
                        const U& d_val) {
  v->assign(n, val);
  d_v->assign(n, d_val);
}

template <typename T, typename U>
void assign_pushforward(::std::vector<T>* v, U first, U last,
                        ::std::vector<T>* d_v, U d_first, U d_last) {
  v->assign(first, last);
  d_v->assign(d_first, d_last);
}

template <typename T, typename U>
void assign_pushforward(::std::vector<T>* v, ::std::initializer_list<U> list,
                        ::std::vector<T>* d_v,
                        ::std::initializer_list<U> d_list) {
  v->assign(list);
  d_v->assign(d_list);
}

template <typename T>
void reserve_pushforward(::std::vector<T>* v,
                         typename ::std::vector<T>::size_type n,
                         ::std::vector<T>* d_v,
                         typename ::std::vector<T>::size_type /*d_n*/) {
  v->reserve(n);
  d_v->reserve(n);
}

template <typename T>
void shrink_to_fit_pushforward(::std::vector<T>* v, ::std::vector<T>* d_v) {
  v->shrink_to_fit();
  d_v->shrink_to_fit();
}

template <typename T, typename U>
void push_back_pushforward(::std::vector<T>* v, U val, ::std::vector<T>* d_v,
                           U d_val) {
  v->push_back(val);
  d_v->push_back(d_val);
}

template <typename T>
void pop_back_pushforward(::std::vector<T>* v, ::std::vector<T>* d_v) noexcept {
  v->pop_back();
  d_v->pop_back();
}

template <typename T>
clad::ValueAndPushforward<::std::size_t, ::std::size_t>
size_pushforward(const ::std::vector<T>* v,
                 const ::std::vector<T>* d_v) noexcept {
  return {v->size(), 0};
}

template <typename T>
clad::ValueAndPushforward<::std::size_t, ::std::size_t>
capacity_pushforward(const ::std::vector<T>* v,
                     const ::std::vector<T>* d_v) noexcept {
  return {v->capacity(), 0};
}

// array forward mode

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

template <typename T, typename U, ::std::size_t N>
void fill_pushforward(::std::array<T, N>* a, const U& u,
                      ::std::array<T, N>* d_a, const U& d_u) {
  a->fill(u);
  d_a->fill(d_u);
}

template <typename T, ::std::size_t N>
clad::ValueAndPushforward<::std::size_t, ::std::size_t>
size_pushforward(const ::std::array<T, N>* a,
                 const ::std::array<T, N>* d_a) noexcept {
  return {a->size(), 0};
}

// vector reverse mode
// more can be found in tests: test/Gradient/STLCustomDerivatives.C

template <typename T, typename U, typename pU>
void push_back_reverse_forw(::std::vector<T>* v, U val, ::std::vector<T>* d_v,
                            pU /*d_val*/) {
  v->push_back(val);
  d_v->push_back(0);
}

template <typename T, typename U, typename pU>
void push_back_pullback(::std::vector<T>* v, U val, ::std::vector<T>* d_v,
                        pU* d_val) {
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
void operator_subscript_pullback(const ::std::vector<T>* vec,
                                 typename ::std::vector<T>::size_type idx,
                                 P d_y, ::std::vector<T>* d_vec,
                                 typename ::std::vector<T>::size_type* d_idx) {
  (*d_vec)[idx] += d_y;
}

template <typename T, typename P>
void operator_subscript_pullback(::std::vector<T>* vec,
                                 typename ::std::vector<T>::size_type idx,
                                 P d_y, ::std::vector<T>* d_vec,
                                 typename ::std::vector<T>::size_type* d_idx) {
  (*d_vec)[idx] += d_y;
}

template <typename T>
clad::ValueAndAdjoint<T&, T&>
at_reverse_forw(::std::vector<T>* vec, typename ::std::vector<T>::size_type idx,
                ::std::vector<T>* d_vec,
                typename ::std::vector<T>::size_type d_idx) {
  return {(*vec)[idx], (*d_vec)[idx]};
}

template <typename T, typename P>
void at_pullback(const ::std::vector<T>* vec,
                 typename ::std::vector<T>::size_type idx, P d_y,
                 ::std::vector<T>* d_vec,
                 typename ::std::vector<T>::size_type* d_idx) {
  (*d_vec)[idx] += d_y;
}

template <typename T, typename P>
void at_pullback(::std::vector<T>* vec,
                 typename ::std::vector<T>::size_type idx, P d_y,
                 ::std::vector<T>* d_vec,
                 typename ::std::vector<T>::size_type* d_idx) {
  (*d_vec)[idx] += d_y;
}

template <typename T, typename U>
void constructor_pullback(
    typename ::std::vector<T>::size_type count, const U& val,
    const typename ::std::vector<T>::allocator_type& alloc,
    ::std::vector<T>* d_this, typename ::std::vector<T>::size_type* d_count,
    U* d_val, typename ::std::vector<T>::allocator_type* d_alloc) {
  for (unsigned i = 0; i < count; ++i)
    *d_val += (*d_this)[i];
  d_this->clear();
}

// A specialization for std::initializer_list (which is replaced with
// clad::array).
template <typename T>
void constructor_pullback(
    clad::array<T> init, const typename ::std::vector<T>::allocator_type& alloc,
    ::std::vector<T>* d_this, clad::array<T>* d_init,
    typename ::std::vector<T>::allocator_type* d_alloc) {
  for (unsigned i = 0; i < init.size(); ++i) {
    (*d_init)[i] += (*d_this)[i];
    (*d_this)[i] = 0;
  }
}

// A specialization for std::initializer_list (which is replaced with
// clad::array).
template <typename T>
void constructor_pullback(clad::array<T> init, ::std::vector<T>* d_this,
                          clad::array<T>* d_init) {
  for (unsigned i = 0; i < init.size(); ++i) {
    (*d_init)[i] += (*d_this)[i];
    (*d_this)[i] = 0;
  }
}

template <typename T, typename U, typename dU>
void assign_pullback(::std::vector<T>* v,
                     typename ::std::vector<T>::size_type n, U /*val*/,
                     ::std::vector<T>* d_v,
                     typename ::std::vector<T>::size_type* /*d_n*/, dU* d_val) {
  for (typename ::std::vector<T>::size_type i = 0; i < n; ++i) {
    (*d_val) += (*d_v)[i];
    (*d_v)[i] = 0;
  }
}

template <typename T>
void reserve_pullback(::std::vector<T>* v,
                      typename ::std::vector<T>::size_type n,
                      ::std::vector<T>* d_v,
                      typename ::std::vector<T>::size_type* /*d_n*/) noexcept {}

template <typename T>
void shrink_to_fit_pullback(::std::vector<T>* /*v*/,
                            ::std::vector<T>* /*d_v*/) noexcept {}

template <typename T>
void capacity_pullback(const ::std::vector<T>* /*v*/,
                       ::std::vector<T>* /*d_v*/) noexcept {}

template <typename T, typename U>
void size_pullback(const ::std::vector<T>* /*v*/, U /*d_y*/,
                   ::std::vector<T>* /*d_v*/) noexcept {}

template <typename T, typename U>
void capacity_pullback(const ::std::vector<T>* /*v*/, U /*d_y*/,
                       ::std::vector<T>* /*d_v*/) noexcept {}

// array reverse mode

template <typename T, ::std::size_t N>
clad::ValueAndAdjoint<T&, T&> operator_subscript_reverse_forw(
    ::std::array<T, N>* arr, typename ::std::array<T, N>::size_type idx,
    ::std::array<T, N>* d_arr, typename ::std::array<T, N>::size_type d_idx) {
  return {(*arr)[idx], (*d_arr)[idx]};
}
template <typename T, ::std::size_t N, typename P>
void operator_subscript_pullback(
    const ::std::array<T, N>* arr, typename ::std::array<T, N>::size_type idx,
    P d_y, ::std::array<T, N>* d_arr,
    typename ::std::array<T, N>::size_type* d_idx) {
  (*d_arr)[idx] += d_y;
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
void at_pullback(const ::std::array<T, N>* arr,
                 typename ::std::array<T, N>::size_type idx, P d_y,
                 ::std::array<T, N>* d_arr,
                 typename ::std::array<T, N>::size_type* d_idx) {
  (*d_arr)[idx] += d_y;
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
void back_pullback(const ::std::array<T, N>* arr,
                   typename ::std::array<T, N>::value_type d_u,
                   ::std::array<T, N>* d_arr) noexcept {
  (*d_arr)[d_arr->size() - 1] += d_u;
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
void front_pullback(const ::std::array<T, N>* arr,
                    typename ::std::array<T, N>::value_type d_u,
                    ::std::array<T, N>* d_arr) {
  (*d_arr)[0] += d_u;
}
template <typename T, ::std::size_t N, typename U>
void size_pullback(const ::std::array<T, N>* /*a*/, U /*d_y*/,
                   ::std::array<T, N>* /*d_a*/) noexcept {}
template <typename T, ::std::size_t N>
void constructor_pullback(const ::std::array<T, N>& arr,
                          ::std::array<T, N>* d_this,
                          ::std::array<T, N>* d_arr) {
  for (size_t i = 0; i < N; ++i)
    (*d_arr)[i] += (*d_this)[i];
}

// tuple forward mode

template <typename... Args1, typename... Args2>
clad::ValueAndPushforward<::std::tuple<Args1...>&, ::std::tuple<Args1...>&>
operator_equal_pushforward(::std::tuple<Args1...>* tu,
                           ::std::tuple<Args2...>&& in,
                           ::std::tuple<Args1...>* d_tu,
                           ::std::tuple<Args2...>&& d_in) noexcept {
  ::std::tuple<Args1...>& t1 = (*tu = in);
  ::std::tuple<Args1...>& t2 = (*d_tu = d_in);
  return {t1, t2};
}

// std::unique_ptr<T> custom derivatives...
template <typename T, typename U>
clad::ValueAndAdjoint<::std::unique_ptr<T>, ::std::unique_ptr<T>>
constructor_reverse_forw(clad::ConstructorReverseForwTag<::std::unique_ptr<T>>,
                         U* p, U* d_p) {
  return {::std::unique_ptr<T>(p), ::std::unique_ptr<T>(d_p)};
}

template <typename T>
clad::ValueAndAdjoint<T&, T&>
operator_star_reverse_forw(::std::unique_ptr<T>* u, ::std::unique_ptr<T>* d_u) {
  return {**u, **d_u};
}

template <typename T, typename U>
void operator_star_pullback(const ::std::unique_ptr<T>* u, U pullback,
                            ::std::unique_ptr<T>* d_u) {
  **d_u += pullback;
}

} // namespace class_functions

namespace std {

// tie and maketuple forward mode

// Helper functions for selecting subtuples
template <::std::size_t shift_amount, ::std::size_t... Is>
constexpr auto shift_sequence(IndexSequence<Is...>) {
  return IndexSequence<shift_amount + Is...>{};
}

template <typename Tuple, ::std::size_t... Indices>
auto select_tuple_elements(const Tuple& tpl, IndexSequence<Indices...>) {
  return ::std::make_tuple(::std::get<Indices>(tpl)...);
}

template <typename Tuple> auto first_half_tuple(const Tuple& tpl) {
  // static_assert(::std::tuple_size<Tuple>::value % 2 == 0);
  constexpr ::std::size_t half = ::std::tuple_size<Tuple>::value / 2;

  constexpr MakeIndexSequence<half> first_half;
  return select_tuple_elements(tpl, first_half);
}

template <typename Tuple> auto second_half_tuple(const Tuple& tpl) {
  // static_assert(::std::tuple_size<Tuple>::value % 2 == 0);
  constexpr ::std::size_t half = ::std::tuple_size<Tuple>::value / 2;

  constexpr MakeIndexSequence<half> first_half;
  constexpr auto second_half = shift_sequence<half>(first_half);
  return select_tuple_elements(tpl, second_half);
}

template <typename Tuple, ::std::size_t... Indices>
auto select_tuple_elements_tie(const Tuple& tpl, IndexSequence<Indices...>) {
  return ::std::tie(::std::get<Indices>(tpl)...);
}

template <typename Tuple> auto first_half_tuple_tie(const Tuple& tpl) {
  // static_assert(::std::tuple_size<Tuple>::value % 2 == 0);
  constexpr ::std::size_t half = ::std::tuple_size<Tuple>::value / 2;

  constexpr MakeIndexSequence<half> first_half;
  return select_tuple_elements_tie(tpl, first_half);
}

template <typename Tuple> auto second_half_tuple_tie(const Tuple& tpl) {
  // static_assert(::std::tuple_size<Tuple>::value % 2 == 0);
  constexpr ::std::size_t half = ::std::tuple_size<Tuple>::value / 2;

  constexpr MakeIndexSequence<half> first_half;
  constexpr auto second_half = shift_sequence<half>(first_half);
  return select_tuple_elements_tie(tpl, second_half);
}

template <typename... Args> auto tie_pushforward(Args&&... args) noexcept {
  ::std::tuple<Args&...> t = ::std::tie(args...);
  return clad::make_value_and_pushforward(first_half_tuple_tie(t),
                                          second_half_tuple_tie(t));
}

template <typename... Args> auto make_tuple_pushforward(Args... args) noexcept {
  ::std::tuple<Args...> t = ::std::make_tuple(args...);
  return clad::make_value_and_pushforward(first_half_tuple(t),
                                          second_half_tuple(t));
}

} // namespace std

} // namespace custom_derivatives
} // namespace clad

#endif // CLAD_STL_BUILTINS_H
