// This header file contains the implementation of the Kokkos framework
// differentiation support in Clad in the form of custom pushforwards and
// pullbacks. Please include it manually to enable Clad for Kokkos code.

#ifndef CLAD_DIFFERENTIATOR_KOKKOSBUILTINS_H
#define CLAD_DIFFERENTIATOR_KOKKOSBUILTINS_H

#include <Kokkos_Core.hpp>
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
    const ::std::string& d_name, const size_t& d_idx0, const size_t& d_idx1,
    const size_t& d_idx2, const size_t& d_idx3, const size_t& d_idx4,
    const size_t& d_idx5, const size_t& d_idx6, const size_t& d_idx7) {
  return {Kokkos::View<DataType, ViewParams...>(name, idx0, idx1, idx2, idx3,
                                                idx4, idx5, idx6, idx7),
          Kokkos::View<DataType, ViewParams...>(
              "_diff_" + name, idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7)};
}

/// View indexing
template <typename View, typename Idx>
inline clad::ValueAndPushforward<typename View::reference_type,
                                 typename View::reference_type>
operator_call_pushforward(const View* v, Idx i0, const View* d_v,
                          Idx /*d_i0*/) {
  return {(*v)(i0), (*d_v)(i0)};
}
template <typename View, typename Idx>
clad::ValueAndPushforward<typename View::reference_type,
                          typename View::reference_type>
operator_call_pushforward(const View* v, Idx i0, Idx i1, const View* d_v,
                          Idx /*d_i0*/, Idx /*d_i1*/) {
  return {(*v)(i0, i1), (*d_v)(i0, i1)};
}
template <typename View, typename Idx>
clad::ValueAndPushforward<typename View::reference_type,
                          typename View::reference_type>
operator_call_pushforward(const View* v, Idx i0, Idx i1, Idx i2,
                          const View* d_v, Idx /*d_i0*/, Idx /*d_i1*/,
                          Idx /*d_i2*/) {
  return {(*v)(i0, i1, i2), (*d_v)(i0, i1, i2)};
}
template <typename View, typename Idx>
clad::ValueAndPushforward<typename View::reference_type,
                          typename View::reference_type>
operator_call_pushforward(const View* v, Idx i0, Idx i1, Idx i2, Idx i3,
                          const View* d_v, Idx /*d_i0*/, Idx /*d_i1*/,
                          Idx /*d_i2*/, Idx /*d_i3*/) {
  return {(*v)(i0, i1, i2, i3), (*d_v)(i0, i1, i2, i3)};
}
template <typename View, typename Idx>
clad::ValueAndPushforward<typename View::reference_type,
                          typename View::reference_type>
operator_call_pushforward(const View* v, Idx i0, Idx i1, Idx i2, Idx i3, Idx i4,
                          const View* d_v, Idx /*d_i0*/, Idx /*d_i1*/,
                          Idx /*d_i2*/, Idx /*d_i3*/, Idx /*d_i4*/) {
  return {(*v)(i0, i1, i2, i3, i4), (*d_v)(i0, i1, i2, i3, i4)};
}
template <typename View, typename Idx>
clad::ValueAndPushforward<typename View::reference_type,
                          typename View::reference_type>
operator_call_pushforward(const View* v, Idx i0, Idx i1, Idx i2, Idx i3, Idx i4,
                          Idx i5, const View* d_v, Idx /*d_i0*/, Idx /*d_i1*/,
                          Idx /*d_i2*/, Idx /*d_i3*/, Idx /*d_i4*/,
                          Idx /*d_i5*/) {
  return {(*v)(i0, i1, i2, i3, i4, i5), (*d_v)(i0, i1, i2, i3, i4, i5)};
}
template <typename View, typename Idx>
clad::ValueAndPushforward<typename View::reference_type,
                          typename View::reference_type>
operator_call_pushforward(const View* v, Idx i0, Idx i1, Idx i2, Idx i3, Idx i4,
                          Idx i5, Idx i6, const View* d_v, Idx /*d_i0*/,
                          Idx /*d_i1*/, Idx /*d_i2*/, Idx /*d_i3*/,
                          Idx /*d_i4*/, Idx /*d_i5*/, Idx /*d_i6*/) {
  return {(*v)(i0, i1, i2, i3, i4, i5, i6), (*d_v)(i0, i1, i2, i3, i4, i5, i6)};
}
template <typename View, typename Idx>
clad::ValueAndPushforward<typename View::reference_type,
                          typename View::reference_type>
operator_call_pushforward(const View* v, Idx i0, Idx i1, Idx i2, Idx i3, Idx i4,
                          Idx i5, Idx i6, Idx i7, const View* d_v, Idx /*d_i0*/,
                          Idx /*d_i1*/, Idx /*d_i2*/, Idx /*d_i3*/,
                          Idx /*d_i4*/, Idx /*d_i5*/, Idx /*d_i6*/,
                          Idx /*d_i7*/) {
  return {(*v)(i0, i1, i2, i3, i4, i5, i6, i7),
          (*d_v)(i0, i1, i2, i3, i4, i5, i6, i7)};
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

/// Fence
template <typename S> void fence_pushforward(const S& s, const S& /*d_s*/) {
  ::Kokkos::fence(s);
}

/// Parallel for
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
    ::Kokkos::parallel_for("_diff_" + str, policy,
                           [&functor, &d_functor](const T x, auto&&... args) {
                             functor.operator_call_pushforward(
                                 x, args..., &d_functor, &x, 0, 0);
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
    ::Kokkos::parallel_for("_diff_" + str, policy,
                           [&functor, &d_functor](const T x, auto&&... args) {
                             functor.operator_call_pushforward(
                                 x, args..., &d_functor, &x, 0, 0, 0);
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
    ::Kokkos::parallel_for("_diff_" + str, policy,
                           [&functor, &d_functor](const T x, auto&&... args) {
                             functor.operator_call_pushforward(
                                 x, args..., &d_functor, &x, 0, 0, 0, 0);
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
    ::Kokkos::parallel_for("_diff_" + str, policy,
                           [&functor, &d_functor](const T x, auto&&... args) {
                             functor.operator_call_pushforward(
                                 x, args..., &d_functor, &x, 0, 0, 0, 0, 0);
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
    ::Kokkos::parallel_for("_diff_" + str, policy,
                           [&functor, &d_functor](const T x, auto&&... args) {
                             functor.operator_call_pushforward(
                                 x, args..., &d_functor, &x, 0, 0, 0, 0, 0, 0);
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
    ::Kokkos::parallel_for("_diff_" + str, policy,
                           [&functor, &d_functor](const T x, auto&&... args) {
                             functor.operator_call_pushforward(
                                 x, args..., &d_functor, &x, {});
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

} // namespace Kokkos
} // namespace clad::custom_derivatives

#endif // CLAD_DIFFERENTIATOR_KOKKOSBUILTINS_H