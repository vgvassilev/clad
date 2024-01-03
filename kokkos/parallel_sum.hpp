#include <Kokkos_Core.hpp>

namespace kokkos_builtin_derivative {

/* Things to do:

- use span_is_contiguous corner case (regardless of the rank)
- check the span of the thing, do we need more than int32.
- deduce iterate base on layout: done?
- If you give me an execution space: non-blocking (in theory) (use an unmaged view if scalar argument)
- If no execution space: blocking.
*/

// Parallel sum:

template <class Viewtype, class Layout, int Rank = Viewtype::rank(), typename iType = int>
struct ViewSum;

template <class Viewtype, class Layout, typename iType>
struct ViewSum<Viewtype, Layout, 1, iType> {

  template<class ExecSpace, class ResultT>
  static void execute(ResultT& result, const Viewtype& v, const ExecSpace space = ExecSpace()) {

    using policy_type = Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<iType>>;
    using value_type  = typename Viewtype::value_type;

    value_type sum;

    Kokkos::parallel_reduce(
        "ViewSum-1D",
        policy_type(space, 0, v.extent(0)),
        KOKKOS_LAMBDA (
            const iType& i0, 
            value_type& update) {
                update += v(i0);
        },
        sum );
    
    result += sum;
  }
};

template <class Viewtype, class Layout, typename iType>
struct ViewSum<Viewtype, Layout, 2, iType> {

  template<class ExecSpace, class ResultT>
  static void execute(ResultT& result, const Viewtype& v, const ExecSpace space = ExecSpace()) {

    static const Kokkos::Iterate outer_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
    static const Kokkos::Iterate inner_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
    using iterate_type =
        Kokkos::Rank<2, outer_iteration_pattern, inner_iteration_pattern>;
    using policy_type =
        Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;
    using value_type  = typename Viewtype::value_type;

    value_type sum;

    Kokkos::parallel_reduce(
        "ViewSum-2D",
        policy_type(space, {0, 0}, {v.extent(0), v.extent(1)}),
        KOKKOS_LAMBDA (
            const iType& i0, 
            const iType& i1, 
            value_type& update) {
                update += v(i0, i1);
        },
        sum );
    
    result += sum;
  }
};

template <class Viewtype, class Layout, typename iType>
struct ViewSum<Viewtype, Layout, 3, iType> {

  template<class ExecSpace, class ResultT>
  static void execute(ResultT& result, const Viewtype& v, const ExecSpace space = ExecSpace()) {

    static const Kokkos::Iterate outer_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
    static const Kokkos::Iterate inner_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
    using iterate_type =
        Kokkos::Rank<3, outer_iteration_pattern, inner_iteration_pattern>;
    using policy_type =
        Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;
    using value_type  = typename Viewtype::value_type;

    value_type sum;

    Kokkos::parallel_reduce(
        "ViewSum-3D",
        policy_type(space, {0, 0}, {v.extent(0), v.extent(1), v.extent(2)}),
        KOKKOS_LAMBDA (
            const iType& i0, 
            const iType& i1,
            const iType& i2, 
            value_type& update) {
                update += v(i0, i1, i2);
        },
        sum );
    
    result += sum;
  }
};


template <typename ViewtypeA>
void parallel_sum(typename ViewtypeA::value_type &sum, const ViewtypeA A) {
  ViewSum<ViewtypeA, typename ViewtypeA::array_layout, ViewtypeA::rank, int>::template execute<typename ViewtypeA::execution_space>(sum, A);
}


// Parallel add

template <class Viewtype, class Layout, int Rank = Viewtype::rank(), typename iType = int>
struct ViewAdd;

template <class Viewtype, class Layout, typename iType>
struct ViewAdd<Viewtype, Layout, 1, iType> {

  template<class ExecSpace, class ResultT>
  static void execute(const Viewtype& v, ResultT& update, const ExecSpace space = ExecSpace()) {

    using policy_type = Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<iType>>;

    Kokkos::parallel_for(
        "ViewAdd-1D",
        policy_type(space, 0, v.extent(0)),
        KOKKOS_LAMBDA (
            const iType& i0) {
                v(i0) += update;
        });
  }
};

template <class Viewtype, class Layout, typename iType>
struct ViewAdd<Viewtype, Layout, 2, iType> {

  template<class ExecSpace, class ResultT>
  static void execute(const Viewtype& v, ResultT& update, const ExecSpace space = ExecSpace()) {

    static const Kokkos::Iterate outer_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
    static const Kokkos::Iterate inner_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
    using iterate_type =
        Kokkos::Rank<2, outer_iteration_pattern, inner_iteration_pattern>;
    using policy_type =
        Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

    Kokkos::parallel_for(
        "ViewAdd-2D",
        policy_type(space, {0, 0}, {v.extent(0), v.extent(1)}),
        KOKKOS_LAMBDA (
            const iType& i0, 
            const iType& i1) {
                v(i0, i1) += update;
        });
  }
};

template <class Viewtype, class Layout, typename iType>
struct ViewAdd<Viewtype, Layout, 3, iType> {

  template<class ExecSpace, class ResultT>
  static void execute(const Viewtype& v, ResultT& update, const ExecSpace space = ExecSpace()) {

    static const Kokkos::Iterate outer_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::outer_iteration_pattern;
    static const Kokkos::Iterate inner_iteration_pattern =
        Kokkos::layout_iterate_type_selector<Layout>::inner_iteration_pattern;
    using iterate_type =
        Kokkos::Rank<3, outer_iteration_pattern, inner_iteration_pattern>;
    using policy_type =
        Kokkos::MDRangePolicy<ExecSpace, iterate_type, Kokkos::IndexType<iType>>;

    Kokkos::parallel_for(
        "ViewAdd-3D",
        policy_type(space, {0, 0}, {v.extent(0), v.extent(1), v.extent(2)}),
        KOKKOS_LAMBDA (
            const iType& i0, 
            const iType& i1,
            const iType& i2) {
                v(i0, i1, i2) += update;
        });
  }
};



template <typename ViewtypeA>
void parallel_sum(ViewtypeA A, const typename ViewtypeA::value_type b) {
  ViewAdd<ViewtypeA, typename ViewtypeA::array_layout, ViewtypeA::rank, int>::template execute<typename ViewtypeA::execution_space>(A, b);
}

}