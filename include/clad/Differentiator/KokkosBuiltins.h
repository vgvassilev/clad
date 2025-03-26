// This header file contains the implementation of the Kokkos framework
// differentiation support in Clad in the form of custom pushforwards and
// pullbacks. Please include it manually to enable Clad for Kokkos code.

#ifndef CLAD_DIFFERENTIATOR_KOKKOSBUILTINS_H
#define CLAD_DIFFERENTIATOR_KOKKOSBUILTINS_H

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <string>
#include <type_traits>
#include "clad/Differentiator/Differentiator.h"

namespace clad::custom_derivatives {
namespace class_functions {
/// Kokkos arrays
template <class DataType, class... ViewParams>
clad::ValueAndPushforward<Kokkos::View<DataType, ViewParams...>,
                          Kokkos::View<DataType, ViewParams...>>
constructor_pushforward(
    clad::ConstructorPushforwardTag<Kokkos::View<DataType, ViewParams...>>,
    const ::std::string& name, const size_t& idx0, const size_t& idx1,
    const size_t& idx2, const size_t& idx3, const size_t& idx4,
    const size_t& idx5, const size_t& idx6, const size_t& idx7,
    const ::std::string& /*d_name*/, const size_t& /*d_idx0*/,
    const size_t& /*d_idx1*/, const size_t& /*d_idx2*/,
    const size_t& /*d_idx3*/, const size_t& /*d_idx4*/,
    const size_t& /*d_idx5*/, const size_t& /*d_idx6*/,
    const size_t& /*d_idx7*/) {
  return {Kokkos::View<DataType, ViewParams...>(name, idx0, idx1, idx2, idx3,
                                                idx4, idx5, idx6, idx7),
          Kokkos::View<DataType, ViewParams...>(
              "_diff_" + name, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7)};
}
template <class DataType, class... ViewParams>
clad::ValueAndAdjoint<::Kokkos::View<DataType, ViewParams...>,
                      ::Kokkos::View<DataType, ViewParams...>>
constructor_reverse_forw(
    clad::ConstructorReverseForwTag<::Kokkos::View<DataType, ViewParams...>>,
    const ::std::string& name, const size_t& idx0, const size_t& idx1,
    const size_t& idx2, const size_t& idx3, const size_t& idx4,
    const size_t& idx5, const size_t& idx6, const size_t& idx7,
    const ::std::string& /*d_name*/, const size_t& /*d_idx0*/,
    const size_t& /*d_idx1*/, const size_t& /*d_idx2*/,
    const size_t& /*d_idx3*/, const size_t& /*d_idx4*/,
    const size_t& /*d_idx5*/, const size_t& /*d_idx6*/,
    const size_t& /*d_idx7*/) {
  return {::Kokkos::View<DataType, ViewParams...>(name, idx0, idx1, idx2, idx3,
                                                  idx4, idx5, idx6, idx7),
          ::Kokkos::View<DataType, ViewParams...>(
              "_diff_" + name, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7)};
}
template <class DataType, class... ViewParams>
void constructor_pullback(const ::std::string& name, const size_t& idx0,
                          const size_t& idx1, const size_t& idx2,
                          const size_t& idx3, const size_t& idx4,
                          const size_t& idx5, const size_t& idx6,
                          const size_t& idx7,
                          ::Kokkos::View<DataType, ViewParams...>* d_this,
                          const ::std::string* /*d_name*/,
                          const size_t& /*d_idx0*/, const size_t* /*d_idx1*/,
                          const size_t* /*d_idx2*/, const size_t* /*d_idx3*/,
                          const size_t* /*d_idx4*/, const size_t* /*d_idx5*/,
                          const size_t* /*d_idx6*/, const size_t* /*d_idx7*/) {}

/// View indexing
template <typename View, typename Idx>
inline clad::ValueAndPushforward<typename View::reference_type,
                                 typename View::reference_type>
operator_call_pushforward(const View* v, Idx i0, const View* d_v,
                          Idx /*d_i0*/) {
  return {(*v)(i0), (*d_v)(i0)};
}
template <typename View, typename Idx0, typename Idx1>
clad::ValueAndPushforward<typename View::reference_type,
                          typename View::reference_type>
operator_call_pushforward(const View* v, Idx0 i0, Idx1 i1, const View* d_v,
                          Idx0 /*d_i0*/, Idx1 /*d_i1*/) {
  return {(*v)(i0, i1), (*d_v)(i0, i1)};
}
template <typename View, typename Idx0, typename Idx1, typename Idx2>
clad::ValueAndPushforward<typename View::reference_type,
                          typename View::reference_type>
operator_call_pushforward(const View* v, Idx0 i0, Idx1 i1, Idx2 i2,
                          const View* d_v, Idx0 /*d_i0*/, Idx1 /*d_i1*/,
                          Idx2 /*d_i2*/) {
  return {(*v)(i0, i1, i2), (*d_v)(i0, i1, i2)};
}
template <typename View, typename Idx0, typename Idx1, typename Idx2,
          typename Idx3>
clad::ValueAndPushforward<typename View::reference_type,
                          typename View::reference_type>
operator_call_pushforward(const View* v, Idx0 i0, Idx1 i1, Idx2 i2, Idx3 i3,
                          const View* d_v, Idx0 /*d_i0*/, Idx1 /*d_i1*/,
                          Idx2 /*d_i2*/, Idx3 /*d_i3*/) {
  return {(*v)(i0, i1, i2, i3), (*d_v)(i0, i1, i2, i3)};
}
template <typename View, typename Idx0, typename Idx1, typename Idx2,
          typename Idx3, typename Idx4>
clad::ValueAndPushforward<typename View::reference_type,
                          typename View::reference_type>
operator_call_pushforward(const View* v, Idx0 i0, Idx1 i1, Idx2 i2, Idx3 i3,
                          Idx4 i4, const View* d_v, Idx0 /*d_i0*/,
                          Idx1 /*d_i1*/, Idx2 /*d_i2*/, Idx3 /*d_i3*/,
                          Idx4 /*d_i4*/) {
  return {(*v)(i0, i1, i2, i3, i4), (*d_v)(i0, i1, i2, i3, i4)};
}
template <typename View, typename Idx0, typename Idx1, typename Idx2,
          typename Idx3, typename Idx4, typename Idx5>
clad::ValueAndPushforward<typename View::reference_type,
                          typename View::reference_type>
operator_call_pushforward(const View* v, Idx0 i0, Idx1 i1, Idx2 i2, Idx3 i3,
                          Idx4 i4, Idx5 i5, const View* d_v, Idx0 /*d_i0*/,
                          Idx1 /*d_i1*/, Idx2 /*d_i2*/, Idx3 /*d_i3*/,
                          Idx4 /*d_i4*/, Idx5 /*d_i5*/) {
  return {(*v)(i0, i1, i2, i3, i4, i5), (*d_v)(i0, i1, i2, i3, i4, i5)};
}
template <typename View, typename Idx0, typename Idx1, typename Idx2,
          typename Idx3, typename Idx4, typename Idx5, typename Idx6>
clad::ValueAndPushforward<typename View::reference_type,
                          typename View::reference_type>
operator_call_pushforward(const View* v, Idx0 i0, Idx1 i1, Idx2 i2, Idx3 i3,
                          Idx4 i4, Idx5 i5, Idx6 i6, const View* d_v,
                          Idx0 /*d_i0*/, Idx1 /*d_i1*/, Idx2 /*d_i2*/,
                          Idx3 /*d_i3*/, Idx4 /*d_i4*/, Idx5 /*d_i5*/,
                          Idx6 /*d_i6*/) {
  return {(*v)(i0, i1, i2, i3, i4, i5, i6), (*d_v)(i0, i1, i2, i3, i4, i5, i6)};
}
template <typename View, typename Idx0, typename Idx1, typename Idx2,
          typename Idx3, typename Idx4, typename Idx5, typename Idx6,
          typename Idx7>
clad::ValueAndPushforward<typename View::reference_type,
                          typename View::reference_type>
operator_call_pushforward(const View* v, Idx0 i0, Idx1 i1, Idx2 i2, Idx3 i3,
                          Idx4 i4, Idx5 i5, Idx6 i6, Idx7 i7, const View* d_v,
                          Idx0 /*d_i0*/, Idx1 /*d_i1*/, Idx2 /*d_i2*/,
                          Idx3 /*d_i3*/, Idx4 /*d_i4*/, Idx5 /*d_i5*/,
                          Idx6 /*d_i6*/, Idx7 /*d_i7*/) {
  return {(*v)(i0, i1, i2, i3, i4, i5, i6, i7),
          (*d_v)(i0, i1, i2, i3, i4, i5, i6, i7)};
}
template <class DataType, class... ViewParams, typename Idx>
clad::ValueAndAdjoint<
    typename ::Kokkos::View<DataType, ViewParams...>::reference_type&,
    typename ::Kokkos::View<DataType, ViewParams...>::reference_type&>
operator_call_reverse_forw(const ::Kokkos::View<DataType, ViewParams...>* v,
                           Idx i0,
                           const ::Kokkos::View<DataType, ViewParams...>* d_v,
                           Idx /*d_i0*/) {
  return {(*v)(i0), (*d_v)(i0)};
}
template <class DataType, class... ViewParams, typename Diff, typename Idx,
          typename dIdx>
void operator_call_pullback(const ::Kokkos::View<DataType, ViewParams...>* v,
                            Idx i0, Diff d_y,
                            ::Kokkos::View<DataType, ViewParams...>* d_v,
                            dIdx* /*d_i0*/) {
  (*d_v)(i0) += d_y;
}
template <class DataType, class... ViewParams, typename Idx0, typename Idx1>
clad::ValueAndAdjoint<
    typename ::Kokkos::View<DataType, ViewParams...>::reference_type&,
    typename ::Kokkos::View<DataType, ViewParams...>::reference_type&>
operator_call_reverse_forw(const ::Kokkos::View<DataType, ViewParams...>* v,
                           Idx0 i0, Idx1 i1,
                           const ::Kokkos::View<DataType, ViewParams...>* d_v,
                           Idx0 /*d_i0*/, Idx1 /*d_i1*/) {
  return {(*v)(i0, i1), (*d_v)(i0, i1)};
}
template <class DataType, class... ViewParams, typename Diff, typename Idx0,
          typename dIdx0, typename Idx1, typename dIdx1>
void operator_call_pullback(const ::Kokkos::View<DataType, ViewParams...>* v,
                            Idx0 i0, Idx1 i1, Diff d_y,
                            ::Kokkos::View<DataType, ViewParams...>* d_v,
                            dIdx0* /*d_i0*/, dIdx1* /*d_i1*/) {
  (*d_v)(i0, i1) += d_y;
}
template <class DataType, class... ViewParams, typename Idx0, typename Idx1,
          typename Idx2>
clad::ValueAndAdjoint<
    typename ::Kokkos::View<DataType, ViewParams...>::reference_type&,
    typename ::Kokkos::View<DataType, ViewParams...>::reference_type&>
operator_call_reverse_forw(const ::Kokkos::View<DataType, ViewParams...>* v,
                           Idx0 i0, Idx1 i1, Idx2 i2,
                           const ::Kokkos::View<DataType, ViewParams...>* d_v,
                           Idx0 /*d_i0*/, Idx1 /*d_i1*/, Idx2 /*d_i2*/) {
  return {(*v)(i0, i1, i2), (*d_v)(i0, i1, i2)};
}
template <class DataType, class... ViewParams, typename Diff, typename Idx0,
          typename dIdx0, typename Idx1, typename dIdx1, typename Idx2,
          typename dIdx2>
void operator_call_pullback(const ::Kokkos::View<DataType, ViewParams...>* v,
                            Idx0 i0, Idx1 i1, Idx2 i2, Diff d_y,
                            ::Kokkos::View<DataType, ViewParams...>* d_v,
                            dIdx0* /*d_i0*/, dIdx1* /*d_i1*/, dIdx2* /*d_i2*/) {
  (*d_v)(i0, i1, i2) += d_y;
}
template <class DataType, class... ViewParams, typename Idx0, typename Idx1,
          typename Idx2, typename Idx3>
clad::ValueAndAdjoint<
    typename ::Kokkos::View<DataType, ViewParams...>::reference_type&,
    typename ::Kokkos::View<DataType, ViewParams...>::reference_type&>
operator_call_reverse_forw(const ::Kokkos::View<DataType, ViewParams...>* v,
                           Idx0 i0, Idx1 i1, Idx2 i2, Idx3 i3,
                           const ::Kokkos::View<DataType, ViewParams...>* d_v,
                           Idx0 /*d_i0*/, Idx1 /*d_i1*/, Idx2 /*d_i2*/,
                           Idx3 /*d_i3*/) {
  return {(*v)(i0, i1, i2, i3), (*d_v)(i0, i1, i2, i3)};
}
template <class DataType, class... ViewParams, typename Diff, typename Idx0,
          typename dIdx0, typename Idx1, typename dIdx1, typename Idx2,
          typename dIdx2, typename Idx3, typename dIdx3>
void operator_call_pullback(const ::Kokkos::View<DataType, ViewParams...>* v,
                            Idx0 i0, Idx1 i1, Idx2 i2, Idx3 i3, Diff d_y,
                            ::Kokkos::View<DataType, ViewParams...>* d_v,
                            dIdx0* /*d_i0*/, dIdx1* /*d_i1*/, dIdx2* /*d_i2*/,
                            dIdx3* /*d_i3*/) {
  (*d_v)(i0, i1, i2, i3) += d_y;
}
template <class DataType, class... ViewParams, typename Idx0, typename Idx1,
          typename Idx2, typename Idx3, typename Idx4>
clad::ValueAndAdjoint<
    typename ::Kokkos::View<DataType, ViewParams...>::reference_type&,
    typename ::Kokkos::View<DataType, ViewParams...>::reference_type&>
operator_call_reverse_forw(const ::Kokkos::View<DataType, ViewParams...>* v,
                           Idx0 i0, Idx1 i1, Idx2 i2, Idx3 i3, Idx4 i4,
                           const ::Kokkos::View<DataType, ViewParams...>* d_v,
                           Idx0 /*d_i0*/, Idx1 /*d_i1*/, Idx2 /*d_i2*/,
                           Idx3 /*d_i3*/, Idx4 /*d_i4*/) {
  return {(*v)(i0, i1, i2, i3, i4), (*d_v)(i0, i1, i2, i3, i4)};
}
template <class DataType, class... ViewParams, typename Diff, typename Idx0,
          typename dIdx0, typename Idx1, typename dIdx1, typename Idx2,
          typename dIdx2, typename Idx3, typename dIdx3, typename Idx4,
          typename dIdx4>
void operator_call_pullback(const ::Kokkos::View<DataType, ViewParams...>* v,
                            Idx0 i0, Idx1 i1, Idx2 i2, Idx3 i3, Idx4 i4,
                            Diff d_y,
                            ::Kokkos::View<DataType, ViewParams...>* d_v,
                            dIdx0* /*d_i0*/, dIdx1* /*d_i1*/, dIdx2* /*d_i2*/,
                            dIdx3* /*d_i3*/, dIdx4* /*d_i4*/) {
  (*d_v)(i0, i1, i2, i3, i4) += d_y;
}
template <class DataType, class... ViewParams, typename Idx0, typename Idx1,
          typename Idx2, typename Idx3, typename Idx4, typename Idx5>
clad::ValueAndAdjoint<
    typename ::Kokkos::View<DataType, ViewParams...>::reference_type&,
    typename ::Kokkos::View<DataType, ViewParams...>::reference_type&>
operator_call_reverse_forw(const ::Kokkos::View<DataType, ViewParams...>* v,
                           Idx0 i0, Idx1 i1, Idx2 i2, Idx3 i3, Idx4 i4, Idx5 i5,
                           const ::Kokkos::View<DataType, ViewParams...>* d_v,
                           Idx0 /*d_i0*/, Idx1 /*d_i1*/, Idx2 /*d_i2*/,
                           Idx3 /*d_i3*/, Idx4 /*d_i4*/, Idx5 /*d_i5*/) {
  return {(*v)(i0, i1, i2, i3, i4, i5), (*d_v)(i0, i1, i2, i3, i4, i5)};
}
template <class DataType, class... ViewParams, typename Diff, typename Idx0,
          typename dIdx0, typename Idx1, typename dIdx1, typename Idx2,
          typename dIdx2, typename Idx3, typename dIdx3, typename Idx4,
          typename dIdx4, typename Idx5, typename dIdx5>
void operator_call_pullback(const ::Kokkos::View<DataType, ViewParams...>* v,
                            Idx0 i0, Idx1 i1, Idx2 i2, Idx3 i3, Idx4 i4,
                            Idx5 i5, Diff d_y,
                            ::Kokkos::View<DataType, ViewParams...>* d_v,
                            dIdx0* /*d_i0*/, dIdx1* /*d_i1*/, dIdx2* /*d_i2*/,
                            dIdx3* /*d_i3*/, dIdx4* /*d_i4*/, dIdx5* /*d_i5*/) {
  (*d_v)(i0, i1, i2, i3, i4, i5) += d_y;
}
template <class DataType, class... ViewParams, typename Idx0, typename Idx1,
          typename Idx2, typename Idx3, typename Idx4, typename Idx5,
          typename Idx6>
clad::ValueAndAdjoint<
    typename ::Kokkos::View<DataType, ViewParams...>::reference_type&,
    typename ::Kokkos::View<DataType, ViewParams...>::reference_type&>
operator_call_reverse_forw(const ::Kokkos::View<DataType, ViewParams...>* v,
                           Idx0 i0, Idx1 i1, Idx2 i2, Idx3 i3, Idx4 i4, Idx5 i5,
                           Idx6 i6,
                           const ::Kokkos::View<DataType, ViewParams...>* d_v,
                           Idx0 /*d_i0*/, Idx1 /*d_i1*/, Idx2 /*d_i2*/,
                           Idx3 /*d_i3*/, Idx4 /*d_i4*/, Idx5 /*d_i5*/,
                           Idx6 /*d_i6*/) {
  return {(*v)(i0, i1, i2, i3, i4, i5, i6), (*d_v)(i0, i1, i2, i3, i4, i5, i6)};
}
template <class DataType, class... ViewParams, typename Diff, typename Idx0,
          typename dIdx0, typename Idx1, typename dIdx1, typename Idx2,
          typename dIdx2, typename Idx3, typename dIdx3, typename Idx4,
          typename dIdx4, typename Idx5, typename dIdx5, typename Idx6,
          typename dIdx6>
void operator_call_pullback(const ::Kokkos::View<DataType, ViewParams...>* v,
                            Idx0 i0, Idx1 i1, Idx2 i2, Idx3 i3, Idx4 i4,
                            Idx5 i5, Idx6 i6, Diff d_y,
                            ::Kokkos::View<DataType, ViewParams...>* d_v,
                            dIdx0* /*d_i0*/, dIdx1* /*d_i1*/, dIdx2* /*d_i2*/,
                            dIdx3* /*d_i3*/, dIdx4* /*d_i3*/, dIdx5* /*d_i3*/,
                            dIdx6* /*d_i3*/) {
  (*d_v)(i0, i1, i2, i3, i4, i5, i6) += d_y;
}
template <class DataType, class... ViewParams, typename Idx0, typename Idx1,
          typename Idx2, typename Idx3, typename Idx4, typename Idx5,
          typename Idx6, typename Idx7>
clad::ValueAndAdjoint<
    typename ::Kokkos::View<DataType, ViewParams...>::reference_type&,
    typename ::Kokkos::View<DataType, ViewParams...>::reference_type&>
operator_call_reverse_forw(const ::Kokkos::View<DataType, ViewParams...>* v,
                           Idx0 i0, Idx1 i1, Idx2 i2, Idx3 i3, Idx4 i4, Idx5 i5,
                           Idx6 i6, Idx7 i7,
                           const ::Kokkos::View<DataType, ViewParams...>* d_v,
                           Idx0 /*d_i0*/, Idx1 /*d_i1*/, Idx2 /*d_i2*/,
                           Idx3 /*d_i3*/, Idx4 /*d_i4*/, Idx5 /*d_i5*/,
                           Idx6 /*d_i6*/, Idx7 /*d_i7*/) {
  return {(*v)(i0, i1, i2, i3, i4, i5, i6, i7),
          (*d_v)(i0, i1, i2, i3, i4, i5, i6, i7)};
}
template <class DataType, class... ViewParams, typename Diff, typename Idx0,
          typename dIdx0, typename Idx1, typename dIdx1, typename Idx2,
          typename dIdx2, typename Idx3, typename dIdx3, typename Idx4,
          typename dIdx4, typename Idx5, typename dIdx5, typename Idx6,
          typename dIdx6, typename Idx7, typename dIdx7>
void operator_call_pullback(const ::Kokkos::View<DataType, ViewParams...>* v,
                            Idx0 i0, Idx1 i1, Idx2 i2, Idx3 i3, Idx4 i4,
                            Idx5 i5, Idx6 i6, Idx7 i7, Diff d_y,
                            ::Kokkos::View<DataType, ViewParams...>* d_v,
                            dIdx0* /*d_i0*/, dIdx1* /*d_i1*/, dIdx2* /*d_i2*/,
                            dIdx3* /*d_i3*/, dIdx4* /*d_i3*/, dIdx5* /*d_i3*/,
                            dIdx6* /*d_i3*/, dIdx7* /*d_i3*/) {
  (*d_v)(i0, i1, i2, i3, i4, i5, i6, i7) += d_y;
}
} // namespace class_functions

/// Kokkos functions (view utils)
namespace Kokkos {
template <typename View1, typename View2, typename T>
inline void deep_copy_pushforward(const View1& dst, const View2& src, T param,
                                  const View1& d_dst, const View2& d_src,
                                  T d_param) {
  deep_copy(dst, src);
  deep_copy(d_dst, d_src);
}
template <typename View, int Rank> struct iterate_over_all_view_elements {
  template <typename F> static void run(const View& v, F func) {}
};
template <typename View> struct iterate_over_all_view_elements<View, 1> {
  template <typename F> static void run(const View& v, F func) {
    ::Kokkos::parallel_for("iterate_over_all_view_elements", v.extent(0), func);
    ::Kokkos::fence();
  }
};
template <typename View> struct iterate_over_all_view_elements<View, 2> {
  template <typename F> static void run(const View& v, F func) {
    ::Kokkos::parallel_for("iterate_over_all_view_elements",
                           ::Kokkos::MDRangePolicy<::Kokkos::Rank<2>>(
                               {0, 0}, {v.extent(0), v.extent(1)}),
                           func);
    ::Kokkos::fence();
  }
};
template <typename View> struct iterate_over_all_view_elements<View, 3> {
  template <typename F> static void run(const View& v, F func) {
    ::Kokkos::parallel_for(
        "iterate_over_all_view_elements",
        ::Kokkos::MDRangePolicy<::Kokkos::Rank<3>>(
            {0, 0, 0}, {v.extent(0), v.extent(1), v.extent(2)}),
        func);
    ::Kokkos::fence();
  }
};
template <typename View> struct iterate_over_all_view_elements<View, 4> {
  template <typename F> static void run(const View& v, F func) {
    ::Kokkos::parallel_for(
        "iterate_over_all_view_elements",
        ::Kokkos::MDRangePolicy<::Kokkos::Rank<4>>(
            {0, 0, 0, 0}, {v.extent(0), v.extent(1), v.extent(2), v.extent(3)}),
        func);
    ::Kokkos::fence();
  }
};
template <typename View> struct iterate_over_all_view_elements<View, 5> {
  template <typename F> static void run(const View& v, F func) {
    ::Kokkos::parallel_for(
        "iterate_over_all_view_elements",
        ::Kokkos::MDRangePolicy<::Kokkos::Rank<5>>(
            {0, 0, 0, 0, 0},
            {v.extent(0), v.extent(1), v.extent(2), v.extent(3), v.extent(4)}),
        func);
    ::Kokkos::fence();
  }
};
template <typename View> struct iterate_over_all_view_elements<View, 6> {
  template <typename F> static void run(const View& v, F func) {
    ::Kokkos::parallel_for(
        "iterate_over_all_view_elements",
        ::Kokkos::MDRangePolicy<::Kokkos::Rank<6>>(
            {0, 0, 0, 0, 0, 0}, {v.extent(0), v.extent(1), v.extent(2),
                                 v.extent(3), v.extent(4), v.extent(5)}),
        func);
    ::Kokkos::fence();
  }
};
template <typename View> struct iterate_over_all_view_elements<View, 7> {
  template <typename F> static void run(const View& v, F func) {
    ::Kokkos::parallel_for(
        "iterate_over_all_view_elements",
        ::Kokkos::MDRangePolicy<::Kokkos::Rank<7>>(
            {0, 0, 0, 0, 0, 0, 0},
            {v.extent(0), v.extent(1), v.extent(2), v.extent(3), v.extent(4),
             v.extent(5), v.extent(6)}),
        func);
    ::Kokkos::fence();
  }
};
template <typename... ViewArgs>
void deep_copy_pullback(
    const ::Kokkos::View<ViewArgs...>& dst,
    typename ::Kokkos::ViewTraits<ViewArgs...>::const_value_type& /*value*/,
    ::std::enable_if_t<::std::is_same<
        typename ::Kokkos::ViewTraits<ViewArgs...>::specialize, void>::value>*,
    ::Kokkos::View<ViewArgs...>* d_dst,
    typename ::Kokkos::ViewTraits<ViewArgs...>::value_type* d_value,
    ::std::enable_if_t<
        ::std::is_same<typename ::Kokkos::ViewTraits<ViewArgs...>::specialize,
                       void>::value>*) {
  typename ::Kokkos::ViewTraits<ViewArgs...>::value_type res = 0;

  iterate_over_all_view_elements<
      ::Kokkos::View<ViewArgs...>,
      ::Kokkos::ViewTraits<ViewArgs...>::rank>::run(dst,
                                                    [&res,
                                                     &d_dst](auto&&... args) {
                                                      res += (*d_dst)(args...);
                                                      (*d_dst)(args...) = 0;
                                                    });

  (*d_value) += res;
}
template <typename... ViewArgs1, typename... ViewArgs2>
inline void deep_copy_pullback(
    const ::Kokkos::View<ViewArgs1...>& dst,
    const ::Kokkos::View<ViewArgs2...>& /*src*/,
    ::std::enable_if_t<
        (::std::is_void<
             typename ::Kokkos::ViewTraits<ViewArgs1...>::specialize>::value &&
         ::std::is_void<
             typename ::Kokkos::ViewTraits<ViewArgs2...>::specialize>::value &&
         ((unsigned int)(::Kokkos::ViewTraits<ViewArgs1...>::rank) != 0 ||
          (unsigned int)(::Kokkos::ViewTraits<ViewArgs2...>::rank) != 0))>*,
    ::Kokkos::View<ViewArgs1...>* d_dst, ::Kokkos::View<ViewArgs2...>* d_src,
    ::std::enable_if_t<
        (::std::is_void<
             typename ::Kokkos::ViewTraits<ViewArgs1...>::specialize>::value &&
         ::std::is_void<
             typename ::Kokkos::ViewTraits<ViewArgs2...>::specialize>::value &&
         ((unsigned int)(::Kokkos::ViewTraits<ViewArgs1...>::rank) != 0 ||
          (unsigned int)(::Kokkos::ViewTraits<ViewArgs2...>::rank) != 0))>*) {
  iterate_over_all_view_elements<::Kokkos::View<ViewArgs1...>,
                                 ::Kokkos::ViewTraits<ViewArgs1...>::rank>::
      run(dst, [&d_src, &d_dst](auto&&... args) {
        (*d_src)(args...) += (*d_dst)(args...);
        (*d_dst)(args...) = 0;
      });
}

template <typename View>
void resize_pushforward(
    View& v, const ::std::size_t n0, const ::std::size_t n1,
    const ::std::size_t n2, const ::std::size_t n3, const ::std::size_t n4,
    const ::std::size_t n5, const ::std::size_t n6, const ::std::size_t n7,
    View& d_v, const ::std::size_t /*d_n*/, const ::std::size_t /*d_n*/,
    const ::std::size_t /*d_n*/, const ::std::size_t /*d_n*/,
    const ::std::size_t /*d_n*/, const ::std::size_t /*d_n*/,
    const ::std::size_t /*d_n*/, const ::std::size_t /*d_n*/) {
  ::Kokkos::resize(v, n0, n1, n2, n3, n4, n5, n6, n7);
  ::Kokkos::resize(d_v, n0, n1, n2, n3, n4, n5, n6, n7);
}
template <class I, class dI, class View>
void resize_pushforward(
    const I& arg, View& v, const ::std::size_t n0, const ::std::size_t n1,
    const ::std::size_t n2, const ::std::size_t n3, const ::std::size_t n4,
    const ::std::size_t n5, const ::std::size_t n6, const ::std::size_t n7,
    const dI& /*d_arg*/, View& d_v, const ::std::size_t /*d_n*/,
    const ::std::size_t /*d_n*/, const ::std::size_t /*d_n*/,
    const ::std::size_t /*d_n*/, const ::std::size_t /*d_n*/,
    const ::std::size_t /*d_n*/, const ::std::size_t /*d_n*/,
    const ::std::size_t /*d_n*/) {
  ::Kokkos::resize(arg, v, n0, n1, n2, n3, n4, n5, n6, n7);
  ::Kokkos::resize(arg, d_v, n0, n1, n2, n3, n4, n5, n6, n7);
}
template <class View>
void resize_reverse_forw(
    View& v, const ::std::size_t n0, const ::std::size_t n1,
    const ::std::size_t n2, const ::std::size_t n3, const ::std::size_t n4,
    const ::std::size_t n5, const ::std::size_t n6, const ::std::size_t n7,
    View& d_v, const ::std::size_t /*d_n*/, const ::std::size_t /*d_n*/,
    const ::std::size_t /*d_n*/, const ::std::size_t /*d_n*/,
    const ::std::size_t /*d_n*/, const ::std::size_t /*d_n*/,
    const ::std::size_t /*d_n*/, const ::std::size_t /*d_n*/) {
  ::Kokkos::resize(v, n0, n1, n2, n3, n4, n5, n6, n7);
  ::Kokkos::resize(d_v, n0, n1, n2, n3, n4, n5, n6, n7);
}
template <class I, class dI, class View>
void resize_reverse_forw(
    const I& arg, View& v, const ::std::size_t n0, const ::std::size_t n1,
    const ::std::size_t n2, const ::std::size_t n3, const ::std::size_t n4,
    const ::std::size_t n5, const ::std::size_t n6, const ::std::size_t n7,
    const dI& /*d_arg*/, View& d_v, const ::std::size_t /*d_n*/,
    const ::std::size_t /*d_n*/, const ::std::size_t /*d_n*/,
    const ::std::size_t /*d_n*/, const ::std::size_t /*d_n*/,
    const ::std::size_t /*d_n*/, const ::std::size_t /*d_n*/,
    const ::std::size_t /*d_n*/) {
  ::Kokkos::resize(arg, v, n0, n1, n2, n3, n4, n5, n6, n7);
  ::Kokkos::resize(arg, d_v, n0, n1, n2, n3, n4, n5, n6, n7);
}
template <class... Args> void resize_pullback(Args... /*args*/) {}

/// Fence
template <typename S> void fence_pushforward(const S& s, const S& /*d_s*/) {
  ::Kokkos::fence(s);
}
template <typename... Args> void fence_pullback(Args...) { ::Kokkos::fence(); }

/// Parallel for (forward mode)
template <class... PolicyParams, class FunctorType> // range policy
void parallel_for_pushforward(
    const ::std::string& str,
    const ::Kokkos::RangePolicy<PolicyParams...>& policy,
    const FunctorType& functor, const ::std::string& /*d_str*/,
    const ::Kokkos::RangePolicy<PolicyParams...>& /*d_policy*/,
    const FunctorType& d_functor) {
  ::Kokkos::parallel_for(str, policy, functor);
  ::Kokkos::parallel_for("_diff_" + str, policy,
                         [&functor, &d_functor](const int i) {
                           functor.operator_call_pushforward(i, &d_functor, 0);
                         });
}
// This structure is used to dispatch parallel for pushforward calls based on
// the rank and the work tag of the MDPolicy
template <class Policy, class FunctorType, class T, int Rank>
struct diff_parallel_for_MDP_call_dispatch {
  static void run(const ::std::string& str, const Policy& policy,
                  const FunctorType& functor, const FunctorType& d_functor) {
    assert(false && "Some parallel_for misuse happened during the compilation "
                    "(templates have not been matched properly).");
  }
};
template <class Policy, class FunctorType, class T>
struct diff_parallel_for_MDP_call_dispatch<Policy, FunctorType, T, 2> {
  static void run(const ::std::string& str, const Policy& policy,
                  const FunctorType& functor, const FunctorType& d_functor) {
    ::Kokkos::parallel_for(
        "_diff_" + str, policy,
        [&functor, &d_functor](const auto x, auto&&... args) {
          functor.operator_call_pushforward(x, args..., &d_functor, {}, 0, 0);
        });
  }
};
template <class Policy, class FunctorType>
struct diff_parallel_for_MDP_call_dispatch<Policy, FunctorType, void, 2> {
  static void run(const ::std::string& str, const Policy& policy,
                  const FunctorType& functor, const FunctorType& d_functor) {
    ::Kokkos::parallel_for(
        "_diff_" + str, policy, [&functor, &d_functor](auto&&... args) {
          functor.operator_call_pushforward(args..., &d_functor, 0, 0);
        });
  }
};
template <class Policy, class FunctorType, class T>
struct diff_parallel_for_MDP_call_dispatch<Policy, FunctorType, T, 3> {
  static void run(const ::std::string& str, const Policy& policy,
                  const FunctorType& functor, const FunctorType& d_functor) {
    ::Kokkos::parallel_for(
        "_diff_" + str, policy,
        [&functor, &d_functor](const auto x, auto&&... args) {
          functor.operator_call_pushforward(x, args..., &d_functor, {}, 0, 0,
                                            0);
        });
  }
};
template <class Policy, class FunctorType>
struct diff_parallel_for_MDP_call_dispatch<Policy, FunctorType, void, 3> {
  static void run(const ::std::string& str, const Policy& policy,
                  const FunctorType& functor, const FunctorType& d_functor) {
    ::Kokkos::parallel_for(
        "_diff_" + str, policy, [&functor, &d_functor](auto&&... args) {
          functor.operator_call_pushforward(args..., &d_functor, 0, 0, 0);
        });
  }
};
template <class Policy, class FunctorType, class T>
struct diff_parallel_for_MDP_call_dispatch<Policy, FunctorType, T, 4> {
  static void run(const ::std::string& str, const Policy& policy,
                  const FunctorType& functor, const FunctorType& d_functor) {
    ::Kokkos::parallel_for(
        "_diff_" + str, policy,
        [&functor, &d_functor](const auto x, auto&&... args) {
          functor.operator_call_pushforward(x, args..., &d_functor, {}, 0, 0, 0,
                                            0);
        });
  }
};
template <class Policy, class FunctorType>
struct diff_parallel_for_MDP_call_dispatch<Policy, FunctorType, void, 4> {
  static void run(const ::std::string& str, const Policy& policy,
                  const FunctorType& functor, const FunctorType& d_functor) {
    ::Kokkos::parallel_for(
        "_diff_" + str, policy, [&functor, &d_functor](auto&&... args) {
          functor.operator_call_pushforward(args..., &d_functor, 0, 0, 0, 0);
        });
  }
};
template <class Policy, class FunctorType, class T>
struct diff_parallel_for_MDP_call_dispatch<Policy, FunctorType, T, 5> {
  static void run(const ::std::string& str, const Policy& policy,
                  const FunctorType& functor, const FunctorType& d_functor) {
    ::Kokkos::parallel_for(
        "_diff_" + str, policy,
        [&functor, &d_functor](const auto x, auto&&... args) {
          functor.operator_call_pushforward(x, args..., &d_functor, {}, 0, 0, 0,
                                            0, 0);
        });
  }
};
template <class Policy, class FunctorType>
struct diff_parallel_for_MDP_call_dispatch<Policy, FunctorType, void, 5> {
  static void run(const ::std::string& str, const Policy& policy,
                  const FunctorType& functor, const FunctorType& d_functor) {
    ::Kokkos::parallel_for(
        "_diff_" + str, policy, [&functor, &d_functor](auto&&... args) {
          functor.operator_call_pushforward(args..., &d_functor, 0, 0, 0, 0, 0);
        });
  }
};
template <class Policy, class FunctorType, class T>
struct diff_parallel_for_MDP_call_dispatch<Policy, FunctorType, T, 6> {
  static void run(const ::std::string& str, const Policy& policy,
                  const FunctorType& functor, const FunctorType& d_functor) {
    ::Kokkos::parallel_for(
        "_diff_" + str, policy,
        [&functor, &d_functor](const auto x, auto&&... args) {
          functor.operator_call_pushforward(x, args..., &d_functor, {}, 0, 0, 0,
                                            0, 0, 0);
        });
  }
};
template <class Policy, class FunctorType>
struct diff_parallel_for_MDP_call_dispatch<Policy, FunctorType, void, 6> {
  static void run(const ::std::string& str, const Policy& policy,
                  const FunctorType& functor, const FunctorType& d_functor) {
    ::Kokkos::parallel_for("_diff_" + str, policy,
                           [&functor, &d_functor](auto&&... args) {
                             functor.operator_call_pushforward(
                                 args..., &d_functor, 0, 0, 0, 0, 0, 0);
                           });
  }
};
template <class PolicyP, class... PolicyParams,
          class FunctorType> // multi-dimensional policy
void parallel_for_pushforward(
    const ::std::string& str,
    const ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>& policy,
    const FunctorType& functor, const ::std::string& /*d_str*/,
    const ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>& /*d_policy*/,
    const FunctorType& d_functor) {
  ::Kokkos::parallel_for(str, policy, functor);
  diff_parallel_for_MDP_call_dispatch<
      ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>, FunctorType,
      typename ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>::work_tag,
      ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>::rank>::run(str, policy,
                                                                    functor,
                                                                    d_functor);
}
// This structure is used to dispatch parallel for pushforward calls based on
// the work tag of other types of policies
template <class Policy, class FunctorType, class T>
struct diff_parallel_for_OP_call_dispatch {
  static void run(const ::std::string& str, const Policy& policy,
                  const FunctorType& functor, const FunctorType& d_functor) {
    ::Kokkos::parallel_for(
        "_diff_" + str, policy,
        [&functor, &d_functor](const auto x, auto&&... args) {
          functor.operator_call_pushforward(x, args..., &d_functor, {}, {});
        });
  }
};
template <class Policy, class FunctorType>
struct diff_parallel_for_OP_call_dispatch<Policy, FunctorType, void> {
  static void run(const ::std::string& str, const Policy& policy,
                  const FunctorType& functor, const FunctorType& d_functor) {
    ::Kokkos::parallel_for(
        "_diff_" + str, policy, [&functor, &d_functor](auto&&... args) {
          functor.operator_call_pushforward(args..., &d_functor, {});
        });
  }
};
// This structure is used to dispatch parallel for pushforward calls for
// integral policies
template <class Policy, class FunctorType, bool IsInt>
struct diff_parallel_for_int_call_dispatch {
  static void run(const ::std::string& str, const Policy& policy,
                  const FunctorType& functor, const FunctorType& d_functor) {
    diff_parallel_for_OP_call_dispatch<
        Policy, FunctorType, typename Policy::work_tag>::run(str, policy,
                                                             functor,
                                                             d_functor);
  }
};
template <class Policy, class FunctorType>
struct diff_parallel_for_int_call_dispatch<Policy, FunctorType, true> {
  static void run(const ::std::string& str, const Policy& policy,
                  const FunctorType& functor, const FunctorType& d_functor) {
    ::Kokkos::parallel_for(
        "_diff_" + str, policy, [&functor, &d_functor](const int i) {
          functor.operator_call_pushforward(i, &d_functor, 0);
        });
  }
};
template <class Policy, class FunctorType> // other policy type
void parallel_for_pushforward(const ::std::string& str, const Policy& policy,
                              const FunctorType& functor,
                              const ::std::string& /*d_str*/,
                              const Policy& /*d_policy*/,
                              const FunctorType& d_functor) {
  ::Kokkos::parallel_for(str, policy, functor);
  diff_parallel_for_int_call_dispatch<
      Policy, FunctorType, ::std::is_integral<Policy>::value>::run(str, policy,
                                                                   functor,
                                                                   d_functor);
}
template <class Policy, class FunctorType> // anonymous loop
void parallel_for_pushforward(const Policy& policy, const FunctorType& functor,
                              const Policy& d_policy,
                              const FunctorType& d_functor) {
  parallel_for_pushforward(::std::string("anonymous_parallel_for"), policy,
                           functor, ::std::string(""), d_policy, d_functor);
}
template <class Policy, class FunctorType> // anonymous loop
void parallel_for_pushforward(
    const Policy& policy, const FunctorType& functor,
    ::std::enable_if_t<::Kokkos::is_execution_policy<Policy>::value>* /*param*/,
    const Policy& d_policy, const FunctorType& d_functor,
    ::std::enable_if_t<
        ::Kokkos::is_execution_policy<Policy>::value>* /*d_param*/) {
  parallel_for_pushforward(::std::string("anonymous_parallel_for"), policy,
                           functor, ::std::string(""), d_policy, d_functor);
}
template <typename Policy, class FunctorType> // alternative signature
void parallel_for_pushforward(Policy policy, const FunctorType& functor,
                              const ::std::string& str, Policy d_policy,
                              const FunctorType& d_functor,
                              const ::std::string& d_str) {
  parallel_for_pushforward(str, policy, functor, d_str, d_policy, d_functor);
}
template <typename Policy, class FunctorType> // alternative signature
void parallel_for_pushforward(
    const Policy& policy, const FunctorType& functor, const ::std::string& str,
    ::std::enable_if_t<::Kokkos::is_execution_policy<Policy>::value>* /*param*/,
    const Policy& d_policy, const FunctorType& d_functor,
    const ::std::string& d_str,
    ::std::enable_if_t<
        ::Kokkos::is_execution_policy<Policy>::value>* /*d_param*/) {
  parallel_for_pushforward(str, policy, functor, d_str, d_policy, d_functor);
}

/// Parallel for (reverse mode)
template <typename F>
void parallel_for_pullback(const size_t work_count, const F& functor,
                           size_t* d_work_count, F* d_functor) {
  // TODO: implement parallel_for pullbacks
}

/// Parallel reduce (forward mode)
// TODO: ADD SUPORT FOR MULTIPLE REDUCED ARGUMENTS
// TODO: ADD SUPPORT FOR UNNAMED LOOPS
// This structure is used to dispatch parallel reduce pushforward calls for
// multidimentional policies
template <class Policy, class FunctorType, class Reduced, class WT, int Rank>
struct diff_parallel_reduce_MDP_dispatch { // non-MDPolicy
  static void run(const ::std::string& str, const Policy& policy,
                  const FunctorType& functor, Reduced& res,
                  const FunctorType& d_functor, Reduced& d_res) {
    ::Kokkos::parallel_reduce(
        "_diff_" + str, policy,
        [&](const auto _work_tag, const auto& i, auto& r, auto& d_r) {
          functor.operator_call_pushforward(_work_tag, i, r, &d_functor, {}, {},
                                            d_r);
        },
        res, d_res);
  }
};
template <class Policy, class FunctorType, class Reduced, int Rank>
struct diff_parallel_reduce_MDP_dispatch<Policy, FunctorType, Reduced, void,
                                         Rank> { // non-MDPolicy
  static void run(const ::std::string& str, const Policy& policy,
                  const FunctorType& functor, Reduced& res,
                  const FunctorType& d_functor, Reduced& d_res) {
    ::Kokkos::parallel_reduce(
        "_diff_" + str, policy,
        [&](const auto& i, auto& r, auto& d_r) {
          functor.operator_call_pushforward(i, r, &d_functor, {}, d_r);
        },
        res, d_res);
  }
};
template <class PolicyP, class... PolicyParams, class FunctorType,
          class Reduced, class WT>
struct diff_parallel_reduce_MDP_dispatch<
    ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>, FunctorType, Reduced, WT,
    2> { // MDPolicy, rank = 2
  static void
  run(const ::std::string& str,
      const ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>& policy,
      const FunctorType& functor, Reduced& res, const FunctorType& d_functor,
      Reduced& d_res) {
    ::Kokkos::parallel_reduce(
        "_diff_" + str, policy,
        [&](const auto _work_tag, const auto& i0, const auto& i1, auto& r,
            auto& d_r) {
          functor.operator_call_pushforward(_work_tag, i0, i1, r, &d_functor,
                                            {}, {}, {}, d_r);
        },
        res, d_res);
  }
};
template <class PolicyP, class... PolicyParams, class FunctorType,
          class Reduced>
struct diff_parallel_reduce_MDP_dispatch<
    ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>, FunctorType, Reduced,
    void, 2> { // MDPolicy, rank = 2
  static void
  run(const ::std::string& str,
      const ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>& policy,
      const FunctorType& functor, Reduced& res, const FunctorType& d_functor,
      Reduced& d_res) {
    ::Kokkos::parallel_reduce(
        "_diff_" + str, policy,
        [&](const auto& i0, const auto& i1, auto& r, auto& d_r) {
          functor.operator_call_pushforward(i0, i1, r, &d_functor, {}, {}, d_r);
        },
        res, d_res);
  }
};
template <class PolicyP, class... PolicyParams, class FunctorType,
          class Reduced, class WT>
struct diff_parallel_reduce_MDP_dispatch<
    ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>, FunctorType, Reduced, WT,
    3> { // MDPolicy, rank = 3
  static void
  run(const ::std::string& str,
      const ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>& policy,
      const FunctorType& functor, Reduced& res, const FunctorType& d_functor,
      Reduced& d_res) {
    ::Kokkos::parallel_reduce(
        "_diff_" + str, policy,
        [&](const auto _work_tag, const auto& i0, const auto& i1,
            const auto& i2, auto& r, auto& d_r) {
          functor.operator_call_pushforward(_work_tag, i0, i1, i2, r,
                                            &d_functor, {}, {}, {}, {}, d_r);
        },
        res, d_res);
  }
};
template <class PolicyP, class... PolicyParams, class FunctorType,
          class Reduced>
struct diff_parallel_reduce_MDP_dispatch<
    ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>, FunctorType, Reduced,
    void, 3> { // MDPolicy, rank = 3
  static void
  run(const ::std::string& str,
      const ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>& policy,
      const FunctorType& functor, Reduced& res, const FunctorType& d_functor,
      Reduced& d_res) {
    ::Kokkos::parallel_reduce(
        "_diff_" + str, policy,
        [&](const auto& i0, const auto& i1, const auto& i2, auto& r,
            auto& d_r) {
          functor.operator_call_pushforward(i0, i1, i2, r, &d_functor, {}, {},
                                            {}, d_r);
        },
        res, d_res);
  }
};
template <class PolicyP, class... PolicyParams, class FunctorType,
          class Reduced, class WT>
struct diff_parallel_reduce_MDP_dispatch<
    ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>, FunctorType, Reduced, WT,
    4> { // MDPolicy, rank = 4
  static void
  run(const ::std::string& str,
      const ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>& policy,
      const FunctorType& functor, Reduced& res, const FunctorType& d_functor,
      Reduced& d_res) {
    ::Kokkos::parallel_reduce(
        "_diff_" + str, policy,
        [&](const auto wt, const auto& i0, const auto& i1, const auto& i2,
            const auto& i3, auto& r, auto& d_r) {
          functor.operator_call_pushforward(wt, i0, i1, i2, i3, r, &d_functor,
                                            {}, {}, {}, {}, {}, d_r);
        },
        res, d_res);
  }
};
template <class PolicyP, class... PolicyParams, class FunctorType,
          class Reduced>
struct diff_parallel_reduce_MDP_dispatch<
    ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>, FunctorType, Reduced,
    void, 4> { // MDPolicy, rank = 4
  static void
  run(const ::std::string& str,
      const ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>& policy,
      const FunctorType& functor, Reduced& res, const FunctorType& d_functor,
      Reduced& d_res) {
    ::Kokkos::parallel_reduce(
        "_diff_" + str, policy,
        [&](const auto& i0, const auto& i1, const auto& i2, const auto& i3,
            auto& r, auto& d_r) {
          functor.operator_call_pushforward(i0, i1, i2, i3, r, &d_functor, {},
                                            {}, {}, {}, d_r);
        },
        res, d_res);
  }
};
template <class PolicyP, class... PolicyParams, class FunctorType,
          class Reduced, class WT>
struct diff_parallel_reduce_MDP_dispatch<
    ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>, FunctorType, Reduced, WT,
    5> { // MDPolicy, rank = 5
  static void
  run(const ::std::string& str,
      const ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>& policy,
      const FunctorType& functor, Reduced& res, const FunctorType& d_functor,
      Reduced& d_res) {
    ::Kokkos::parallel_reduce(
        "_diff_" + str, policy,
        [&](const auto wt, const auto& i0, const auto& i1, const auto& i2,
            const auto& i3, const auto& i4, auto& r, auto& d_r) {
          functor.operator_call_pushforward(wt, i0, i1, i2, i3, i4, r,
                                            &d_functor, {}, {}, {}, {}, {}, {},
                                            d_r);
        },
        res, d_res);
  }
};
template <class PolicyP, class... PolicyParams, class FunctorType,
          class Reduced>
struct diff_parallel_reduce_MDP_dispatch<
    ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>, FunctorType, Reduced,
    void, 5> { // MDPolicy, rank = 5
  static void
  run(const ::std::string& str,
      const ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>& policy,
      const FunctorType& functor, Reduced& res, const FunctorType& d_functor,
      Reduced& d_res) {
    ::Kokkos::parallel_reduce(
        "_diff_" + str, policy,
        [&](const auto& i0, const auto& i1, const auto& i2, const auto& i3,
            const auto& i4, auto& r, auto& d_r) {
          functor.operator_call_pushforward(i0, i1, i2, i3, i4, r, &d_functor,
                                            {}, {}, {}, {}, {}, d_r);
        },
        res, d_res);
  }
};
template <class PolicyP, class... PolicyParams, class FunctorType,
          class Reduced, class WT>
struct diff_parallel_reduce_MDP_dispatch<
    ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>, FunctorType, Reduced, WT,
    6> { // MDPolicy, rank = 6
  static void
  run(const ::std::string& str,
      const ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>& policy,
      const FunctorType& functor, Reduced& res, const FunctorType& d_functor,
      Reduced& d_res) {
    ::Kokkos::parallel_reduce(
        "_diff_" + str, policy,
        [&](const auto wt, const auto& i0, const auto& i1, const auto& i2,
            const auto& i3, const auto& i4, const auto& i5, auto& r,
            auto& d_r) {
          functor.operator_call_pushforward(wt, i0, i1, i2, i3, i4, i5, r,
                                            &d_functor, {}, {}, {}, {}, {}, {},
                                            {}, d_r);
        },
        res, d_res);
  }
};
template <class PolicyP, class... PolicyParams, class FunctorType,
          class Reduced>
struct diff_parallel_reduce_MDP_dispatch<
    ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>, FunctorType, Reduced,
    void, 6> { // MDPolicy, rank = 6
  static void
  run(const ::std::string& str,
      const ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>& policy,
      const FunctorType& functor, Reduced& res, const FunctorType& d_functor,
      Reduced& d_res) {
    ::Kokkos::parallel_reduce(
        "_diff_" + str, policy,
        [&](const auto& i0, const auto& i1, const auto& i2, const auto& i3,
            const auto& i4, const auto& i5, auto& r, auto& d_r) {
          functor.operator_call_pushforward(i0, i1, i2, i3, i4, i5, r,
                                            &d_functor, {}, {}, {}, {}, {}, {},
                                            d_r);
        },
        res, d_res);
  }
};
template <class PolicyP, class... PolicyParams, class FunctorType,
          class Reduced>
struct diff_parallel_reduce_MDP_dispatch<
    ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>, FunctorType, Reduced,
    void, 0> { // MDPolicy matched, now figure out the rank
  static void
  run(const ::std::string& str,
      const ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>& policy,
      const FunctorType& functor, Reduced& res, const FunctorType& d_functor,
      Reduced& d_res) {
    diff_parallel_reduce_MDP_dispatch<
        ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>, FunctorType, Reduced,
        typename ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>::work_tag,
        ::Kokkos::MDRangePolicy<PolicyP, PolicyParams...>::rank>::run(str,
                                                                      policy,
                                                                      functor,
                                                                      res,
                                                                      d_functor,
                                                                      d_res);
  }
};
// This structure is used to dispatch parallel reduce pushforward calls for
// integral policies
template <class Policy, class FunctorType, class Reduced, bool isInt>
struct diff_parallel_reduce_int_dispatch { // non-integral policy
  static void run(const ::std::string& str, const Policy& policy,
                  const FunctorType& functor, Reduced& res,
                  const FunctorType& d_functor, Reduced& d_res) {
    diff_parallel_reduce_MDP_dispatch<Policy, FunctorType, Reduced, void,
                                      0>::run(str, policy, functor, res,
                                              d_functor, d_res);
  }
};
template <class Policy, class FunctorType, class Reduced> // integral policy
struct diff_parallel_reduce_int_dispatch<Policy, FunctorType, Reduced, true> {
  static void run(const ::std::string& str, const Policy& policy,
                  const FunctorType& functor, Reduced& res,
                  const FunctorType& d_functor, Reduced& d_res) {
    ::Kokkos::parallel_reduce(
        "_diff_" + str, policy,
        [&](const auto& i, auto& r, auto& d_r) {
          functor.operator_call_pushforward(i, r, &d_functor, {}, d_r);
        },
        res, d_res);
  }
};
template <class Policy, class FunctorType,
          class Reduced> // generally, this is matched
void parallel_reduce_pushforward(const ::std::string& str, const Policy& policy,
                                 const FunctorType& functor, Reduced& res,
                                 const ::std::string& /*d_str*/,
                                 const Policy& /*d_policy*/,
                                 const FunctorType& d_functor, Reduced& d_res) {
  diff_parallel_reduce_int_dispatch<
      Policy, FunctorType, Reduced,
      ::std::is_integral<Policy>::value>::run(str, policy, functor, res,
                                              d_functor, d_res);
}

} // namespace Kokkos
} // namespace clad::custom_derivatives

#endif // CLAD_DIFFERENTIATOR_KOKKOSBUILTINS_H