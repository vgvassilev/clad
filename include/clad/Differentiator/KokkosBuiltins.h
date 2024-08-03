// This header file contains the implementation of the Kokkos framework
// differentiation support in Clad in the form of custom pushforwards and
// pullbacks. Please include it manually to enable Clad for Kokkos code.

#ifndef CLAD_DIFFERENTIATOR_KOKKOSBUILTINS_H
#define CLAD_DIFFERENTIATOR_KOKKOSBUILTINS_H

#include <Kokkos_Core.hpp>
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
    const ::std::string& d_name, const size_t& d_idx0, const size_t& d_idx1,
    const size_t& d_idx2, const size_t& d_idx3, const size_t& d_idx4,
    const size_t& d_idx5, const size_t& d_idx6, const size_t& d_idx7) {
  return {Kokkos::View<DataType, ViewParams...>(name, idx0, idx1, idx2, idx3,
                                                idx4, idx5, idx6, idx7),
          Kokkos::View<DataType, ViewParams...>(
              "_diff_" + name, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7)};
}
} // namespace class_functions

/// Kokkos functions
namespace Kokkos {
template <typename View1, typename View2, typename T>
inline void deep_copy_pushforward(const View1& dst, const View2& src, T param,
                                  const View1& d_dst, const View2& d_src,
                                  T d_param) {
  deep_copy(dst, src);
  deep_copy(d_dst, d_src);
}

template <class View>
inline void resize_pushforward(View& v, const size_t n0, const size_t n1,
                               const size_t n2, const size_t n3,
                               const size_t n4, const size_t n5,
                               const size_t n6, const size_t n7, View& d_v,
                               const size_t /*d_n0*/, const size_t /*d_n1*/,
                               const size_t /*d_n2*/, const size_t /*d_n3*/,
                               const size_t /*d_n4*/, const size_t /*d_n5*/,
                               const size_t /*d_n6*/, const size_t /*d_n7*/) {
  ::Kokkos::resize(v, n0, n1, n2, n3, n4, n5, n6, n7);
  ::Kokkos::resize(d_v, n0, n1, n2, n3, n4, n5, n6, n7);
}

template <class I, class dI, class View>
inline void resize_pushforward(const I& arg, View& v, const size_t n0,
                               const size_t n1, const size_t n2,
                               const size_t n3, const size_t n4,
                               const size_t n5, const size_t n6,
                               const size_t n7, const dI& /*d_arg*/, View& d_v,
                               const size_t /*d_n0*/, const size_t /*d_n1*/,
                               const size_t /*d_n2*/, const size_t /*d_n3*/,
                               const size_t /*d_n4*/, const size_t /*d_n5*/,
                               const size_t /*d_n6*/, const size_t /*d_n7*/) {
  ::Kokkos::resize(arg, v, n0, n1, n2, n3, n4, n5, n6, n7);
  ::Kokkos::resize(arg, d_v, n0, n1, n2, n3, n4, n5, n6, n7);
}

template <class ExecPolicy, class FunctorType>
inline void
parallel_for_pushforward(const ::std::string& str, const ExecPolicy& policy,
                         const FunctorType& functor, const ::std::string& d_str,
                         const ExecPolicy& d_policy,
                         const FunctorType& d_functor) {
  // TODO: implement parallel_for_pushforward
  return;
}
} // namespace Kokkos
} // namespace clad::custom_derivatives

#endif // CLAD_DIFFERENTIATOR_KOKKOSBUILTINS_H