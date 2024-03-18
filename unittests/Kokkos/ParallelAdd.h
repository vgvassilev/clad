// source:
// https://github.com/kliegeois/clad/blob/kokkos-PR/unittests/Kokkos/parallel_sum.hpp

#ifndef KOKKOS_UNITTEST_PARALLELSUM
#define KOKKOS_UNITTEST_PARALLELSUM

#include <Kokkos_Core.hpp>

namespace kokkos_builtin_derivative {

// Parallel sum:

template <class Viewtype, class Layout, int Rank = Viewtype::rank(),
          typename iType = int>
struct ViewSum;

template <class Viewtype, class Layout, typename iType>
struct ViewSum<Viewtype, Layout, 1, iType> {

  template <class ExecSpace, class ResultT>
  static void execute(ResultT& result, const Viewtype& v,
                      const ExecSpace space = ExecSpace()) {

    using policy_type =
        Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<iType>>;
    using value_type = typename Viewtype::value_type;

    value_type sum;

    Kokkos::parallel_reduce(
        "ViewSum-1D", policy_type(space, 0, v.extent(0)),
        KOKKOS_LAMBDA(const iType& i0, value_type& update) { update += v(i0); },
        sum);

    result += sum;
  }
};

template <class Viewtype, class Layout, typename iType>
struct ViewSum<Viewtype, Layout, 2, iType> {

  template <class ExecSpace, class ResultT>
  static void execute(ResultT& result, const Viewtype& v,
                      const ExecSpace space = ExecSpace()) {

    static const Kokkos::Iterate outer_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
    static const Kokkos::Iterate inner_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
    using iterate_type =
        Kokkos::Rank<2, outer_iteration_pattern, inner_iteration_pattern>;
    using policy_type = Kokkos::MDRangePolicy<ExecSpace, iterate_type,
                                              Kokkos::IndexType<iType>>;
    using value_type = typename Viewtype::value_type;

    value_type sum;

    Kokkos::parallel_reduce(
        "ViewSum-2D", policy_type(space, {0, 0}, {v.extent(0), v.extent(1)}),
        KOKKOS_LAMBDA(const iType& i0, const iType& i1, value_type& update) {
          update += v(i0, i1);
        },
        sum);

    result += sum;
  }
};

template <class Viewtype, class Layout, typename iType>
struct ViewSum<Viewtype, Layout, 3, iType> {

  template <class ExecSpace, class ResultT>
  static void execute(ResultT& result, const Viewtype& v,
                      const ExecSpace space = ExecSpace()) {

    static const Kokkos::Iterate outer_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
    static const Kokkos::Iterate inner_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
    using iterate_type =
        Kokkos::Rank<3, outer_iteration_pattern, inner_iteration_pattern>;
    using policy_type = Kokkos::MDRangePolicy<ExecSpace, iterate_type,
                                              Kokkos::IndexType<iType>>;
    using value_type = typename Viewtype::value_type;

    value_type sum;

    Kokkos::parallel_reduce(
        "ViewSum-3D",
        policy_type(space, {0, 0}, {v.extent(0), v.extent(1), v.extent(2)}),
        KOKKOS_LAMBDA(const iType& i0, const iType& i1, const iType& i2,
                      value_type& update) { update += v(i0, i1, i2); },
        sum);

    result += sum;
  }
};

// Parallel add

template <class Viewtype, class Layout, int Rank = Viewtype::rank(),
          typename iType = int>
struct ViewAdd;

template <class Viewtype, class Layout, typename iType>
struct ViewAdd<Viewtype, Layout, 1, iType> {

  template <class ExecSpace, class ResultT>
  static void execute(const Viewtype& v, ResultT& update,
                      const ExecSpace space = ExecSpace()) {

    using policy_type =
        Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<iType>>;

    Kokkos::parallel_for(
        "ViewAdd-1D", policy_type(space, 0, v.extent(0)),
        KOKKOS_LAMBDA(const iType& i0) { v(i0) += update; });
  }

  template <class ExecSpace, class ResultT>
  static void executeView(const Viewtype& v, ResultT& update,
                          const ExecSpace space = ExecSpace()) {

    using policy_type =
        Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<iType>>;

    Kokkos::parallel_for(
        "ViewAdd-1D", policy_type(space, 0, v.extent(0)),
        KOKKOS_LAMBDA(const iType& i0) { v(i0) += update(i0); });
  }
};

template <class Viewtype, class Layout, typename iType>
struct ViewAdd<Viewtype, Layout, 2, iType> {

  template <class ExecSpace, class ResultT>
  static void execute(const Viewtype& v, ResultT& update,
                      const ExecSpace space = ExecSpace()) {

    static const Kokkos::Iterate outer_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
    static const Kokkos::Iterate inner_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
    using iterate_type =
        Kokkos::Rank<2, outer_iteration_pattern, inner_iteration_pattern>;
    using policy_type = Kokkos::MDRangePolicy<ExecSpace, iterate_type,
                                              Kokkos::IndexType<iType>>;

    Kokkos::parallel_for(
        "ViewAdd-2D", policy_type(space, {0, 0}, {v.extent(0), v.extent(1)}),
        KOKKOS_LAMBDA(const iType& i0, const iType& i1) {
          v(i0, i1) += update;
        });
  }

  template <class ExecSpace, class ResultT>
  static void executeView(const Viewtype& v, ResultT& update,
                          const ExecSpace space = ExecSpace()) {

    static const Kokkos::Iterate outer_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
    static const Kokkos::Iterate inner_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
    using iterate_type =
        Kokkos::Rank<2, outer_iteration_pattern, inner_iteration_pattern>;
    using policy_type = Kokkos::MDRangePolicy<ExecSpace, iterate_type,
                                              Kokkos::IndexType<iType>>;

    Kokkos::parallel_for(
        "ViewAdd-2D", policy_type(space, {0, 0}, {v.extent(0), v.extent(1)}),
        KOKKOS_LAMBDA(const iType& i0, const iType& i1) {
          v(i0, i1) += update(i0, i1);
        });
  }
};

template <class Viewtype, class Layout, typename iType>
struct ViewAdd<Viewtype, Layout, 3, iType> {

  template <class ExecSpace, class ResultT>
  static void execute(const Viewtype& v, ResultT& update,
                      const ExecSpace space = ExecSpace()) {

    static const Kokkos::Iterate outer_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
    static const Kokkos::Iterate inner_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
    using iterate_type =
        Kokkos::Rank<3, outer_iteration_pattern, inner_iteration_pattern>;
    using policy_type = Kokkos::MDRangePolicy<ExecSpace, iterate_type,
                                              Kokkos::IndexType<iType>>;

    Kokkos::parallel_for(
        "ViewAdd-3D",
        policy_type(space, {0, 0}, {v.extent(0), v.extent(1), v.extent(2)}),
        KOKKOS_LAMBDA(const iType& i0, const iType& i1, const iType& i2) {
          v(i0, i1, i2) += update;
        });
  }

  template <class ExecSpace, class ResultT>
  static void executeView(const Viewtype& v, ResultT& update,
                          const ExecSpace space = ExecSpace()) {

    static const Kokkos::Iterate outer_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
    static const Kokkos::Iterate inner_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
    using iterate_type =
        Kokkos::Rank<3, outer_iteration_pattern, inner_iteration_pattern>;
    using policy_type = Kokkos::MDRangePolicy<ExecSpace, iterate_type,
                                              Kokkos::IndexType<iType>>;

    Kokkos::parallel_for(
        "ViewAdd-3D",
        policy_type(space, {0, 0}, {v.extent(0), v.extent(1), v.extent(2)}),
        KOKKOS_LAMBDA(const iType& i0, const iType& i1, const iType& i2) {
          v(i0, i1, i2) += update(i0, i1, i2);
        });
  }
};

template <class DT, class... DP>
void parallel_sum(typename Kokkos::ViewTraits<DT, DP...>::value_type& sum,
                  const Kokkos::View<DT, DP...> A) {
  using ViewtypeA = Kokkos::View<DT, DP...>;
  Kokkos::fence();
  if (A.span_is_contiguous()) {

    using ViewTypeFlat = Kokkos::View<
        typename ViewtypeA::value_type*, Kokkos::LayoutRight,
        Kokkos::Device<typename ViewtypeA::execution_space,
                       std::conditional_t<ViewtypeA::rank == 0,
                                          typename ViewtypeA::memory_space,
                                          Kokkos::AnonymousSpace>>,
        Kokkos::MemoryTraits<0>>;

    ViewTypeFlat A_flat(A.data(), A.size());
    ViewSum<ViewTypeFlat, Kokkos::LayoutRight, 1, int>::template execute<
        typename ViewTypeFlat::execution_space>(sum, A_flat);
  } else {
    ViewSum<ViewtypeA, typename ViewtypeA::array_layout, ViewtypeA::rank,
            int>::template execute<typename ViewtypeA::execution_space>(sum, A);
  }
  Kokkos::fence();
}

template <class ExecSpace, class DT, class... DP>
void parallel_sum(const ExecSpace& space,
                  typename Kokkos::ViewTraits<DT, DP...>::value_type& sum,
                  const Kokkos::View<DT, DP...> A) {
  using ViewtypeA = Kokkos::View<DT, DP...>;
  space.fence();
  if (A.span_is_contiguous()) {

    using ViewTypeFlat = Kokkos::View<
        typename ViewtypeA::value_type*, Kokkos::LayoutRight,
        Kokkos::Device<typename ViewtypeA::execution_space,
                       std::conditional_t<ViewtypeA::rank == 0,
                                          typename ViewtypeA::memory_space,
                                          Kokkos::AnonymousSpace>>,
        Kokkos::MemoryTraits<0>>;

    ViewTypeFlat A_flat(A.data(), A.size());
    ViewSum<ViewTypeFlat, Kokkos::LayoutRight, 1, int>::template execute<
        typename ViewTypeFlat::execution_space>(sum, A_flat, space);
  } else {
    ViewSum<ViewtypeA, typename ViewtypeA::array_layout, ViewtypeA::rank,
            int>::template execute<ExecSpace>(sum, A, space);
  }
  space.fence();
}

template <class DT, class... DP>
void parallel_add(Kokkos::View<DT, DP...> A,
                  typename Kokkos::ViewTraits<DT, DP...>::const_value_type b) {
  using ViewtypeA = Kokkos::View<DT, DP...>;
  Kokkos::fence();
  if (A.span_is_contiguous()) {

    using ViewTypeFlat = Kokkos::View<
        typename ViewtypeA::value_type*, Kokkos::LayoutRight,
        Kokkos::Device<typename ViewtypeA::execution_space,
                       std::conditional_t<ViewtypeA::rank == 0,
                                          typename ViewtypeA::memory_space,
                                          Kokkos::AnonymousSpace>>,
        Kokkos::MemoryTraits<0>>;

    ViewTypeFlat A_flat(A.data(), A.size());
    ViewAdd<ViewTypeFlat, Kokkos::LayoutRight, 1, int>::template execute<
        typename ViewTypeFlat::execution_space>(A_flat, b);
  } else {
    ViewAdd<ViewtypeA, typename ViewtypeA::array_layout, ViewtypeA::rank,
            int>::template execute<typename ViewtypeA::execution_space>(A, b);
  }
  Kokkos::fence();
}

template <class ExecSpace, class DT, class... DP>
void parallel_add(const ExecSpace& space, Kokkos::View<DT, DP...> A,
                  typename Kokkos::ViewTraits<DT, DP...>::const_value_type b) {
  using ViewtypeA = Kokkos::View<DT, DP...>;
  space.fence();
  if (A.span_is_contiguous()) {

    using ViewTypeFlat = Kokkos::View<
        typename ViewtypeA::value_type*, Kokkos::LayoutRight,
        Kokkos::Device<typename ViewtypeA::execution_space,
                       std::conditional_t<ViewtypeA::rank == 0,
                                          typename ViewtypeA::memory_space,
                                          Kokkos::AnonymousSpace>>,
        Kokkos::MemoryTraits<0>>;

    ViewTypeFlat A_flat(A.data(), A.size());
    ViewAdd<ViewTypeFlat, Kokkos::LayoutRight, 1, int>::template execute<
        typename ViewTypeFlat::execution_space>(A_flat, b, space);
  } else {
    ViewAdd<ViewtypeA, typename ViewtypeA::array_layout, ViewtypeA::rank,
            int>::template execute<ExecSpace>(A, b, space);
  }
  space.fence();
}

template <class DT, class... DP, class ST, class... SP>
void parallel_add(Kokkos::View<DT, DP...> A, const Kokkos::View<ST, SP...> B) {
  using ViewtypeA = Kokkos::View<DT, DP...>;
  using ViewtypeA = Kokkos::View<ST, SP...>;
  Kokkos::fence();

  ViewAdd<ViewtypeA, typename ViewtypeA::array_layout, ViewtypeA::rank,
          int>::template executeView<typename ViewtypeA::execution_space>(A, B);

  Kokkos::fence();
}

} // namespace kokkos_builtin_derivative

#endif