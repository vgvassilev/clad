#ifndef CLAD_NO_NUM_DIFF
#ifndef CLAD_NUMERICAL_DIFF_H
#define CLAD_NUMERICAL_DIFF_H

#include "ArrayRef.h"
#include "FunctionTraits.h"
#include "Tape.h"

#include <cmath>
#include <limits>
#include <memory>
#include <utility>

namespace numerical_diff {

  /// A class to keep track of the memory we allocate to make sure it is
  /// deallocated later.
  class ManageBufferSpace {
    /// A tape of shared pointers so we don't have to do memory management
    /// ourselves.
    clad::tape_impl<void*> data = {};

  public:
    ~ManageBufferSpace() { free_buffer(); }

    /// A function to make some buffer space and construct object in place
    /// if the given type has a trivial destructor and construction is
    /// requested. This provided so that users can numerically differentiate
    /// functions that take pointers to user-defined data types.
    /// FIXME: implement type-erasure to support types with non-trivial
    /// destructors.
    ///
    /// \tparam T the type of buffer to make.
    /// \param[in] \c n The length for the buffer.
    /// \param[in] \c constructInPlace True if an object has to be constructed
    /// in place. Use for trivially constructable types ONLY.
    /// \param[in] \c args Arguments to forward to the constructor.
    ///
    /// \returns A raw pointer to the newly created buffer.
    template <typename T, typename... Args,
              typename std::enable_if<std::is_trivially_destructible<T>::value,
                                      bool>::type = true>
    T* make_buffer_space(std::size_t n, bool constructInPlace, Args&&... args) {
      void* ptr = malloc(n * sizeof(T));
      if (constructInPlace) {
        ::new (ptr) T(std::forward<Args>(args)...);
      }
      data.emplace_back(ptr);
      return static_cast<T*>(ptr);
    }

    /// A function to make some buffer space.
    ///
    /// \tparam T the type of buffer to make.
    /// \param[in] \c n The length for the buffer.
    ///
    /// \returns A raw pointer to the newly created buffer.
    template <typename T> T* make_buffer_space(std::size_t n) {
      void* ptr = malloc(n * sizeof(T));
      data.emplace_back(ptr);
      return static_cast<T*>(ptr);
    }

    /// A function to free the space previously allocated.
    void free_buffer() {
      while (data.size()) {
        void* ptr = data.back();
        data.pop_back();
        free(ptr);
      }
    }
  };

  /// A buffer manager to request for buffer space
  /// while forwarding reference/pointer args to the target function.
  ManageBufferSpace bufferManager;

  /// The precision to do the numerical differentiation calculations in.
  using precision = double;

  /// A function to make sure the step size being used is machine representable.
  /// It is likely that if do not have a similar construct, we may end up with
  /// catastrophic cancellations, hence resulting into a 0 derivative over a
  /// large interval.
  ///
  /// \param[in] \c x Input to the target function.
  /// \param[in] \c h The h to make representable.
  ///
  /// \returns A value of h that does not result in catastrohic cancellation.
  precision make_h_representable(precision x, precision h) {
    precision xph = x + h;
    precision dx = xph - x;

    // If x+h ~ x, we should look for the next closest value so that (x+h) - x
    // != 0
    if (dx == 0)
      h = std::nextafter(x, std::numeric_limits<precision>::max()) - x;

    return h;
  }

  /// A function to return the h value to use.
  ///
  /// \param[in] \c arg The input argument to adjust and get h for.
  ///
  /// \returns A calculated and adjusted h value.
  precision get_h(precision arg) {
    // First get a suitable h value, we do all of this in elevated precision
    // (default double). Maximum error in h = eps^4/5
    precision h = std::pow(11.25 * std::numeric_limits<precision>::epsilon(),
                           (double)1.0 / 5.0);
    h = make_h_representable(arg, h);
    return h;
  }

  /// A function to print the errors associated with numerically differentiating
  /// a function to the standard output stream. This can be enabled by
  /// '-fprint-num-diff-errors'.
  ///
  /// \param[in] \c derivError The error associated with numerically
  /// differentiating a function. This error is calculated by the remainder term
  /// on expanding the five-point stencil series.
  /// \param[in] \c evalError This error is associated with the evaluation of
  /// the target functions.
  /// \param[in] \c paramPos The position of the parameter to which this error
  ///  belongs.
  /// \param[in] \c arrPos The position of the array element
  /// (-1 if parameter is scalar) to which the error belongs.
  void printError(precision derivError, precision evalError, unsigned paramPos,
                  int arrPos = -1) {
    if (arrPos != -1)
      printf("\nError Report for parameter at position %d and index %d:\n",
             paramPos, arrPos);
    else
      printf("\nError Report for parameter at position %d:\n", paramPos);
    printf("Error due to the five-point central difference is: %0.10f"
           "\nError due to function evaluation is: %0.10f\n",
           derivError, evalError);
  }

  /// A function to update scalar parameter values given a multiplier and the
  /// argument value. Custom data types can be supported as input in the case
  /// that this function is overloaded.
  ///
  /// This function should be overloaded as follows:
  /// \code
  /// template <typename <--your-data-type-->>
  /// <--your-data-type-->* updateIndexParamValue(<--your-data-type--> arg,
  ///                           std::size_t idx, std::size_t currIdx,
  ///                          int multiplier, precision& h_val,
  ///                          std::size_t n = 0, std::size_t i = 0) {
  ///   if (idx == currIdx) {
  ///     // Add your type specific code here that would essentially translate
  ///     // to `return arg + multiplier * h` (h is some very small value)
  ///     h_val = //Save the 'precision' equivalent value of h here only if it
  ///     is zero, not doing so will lead to undefined behaviour.
  ///   }
  ///   return arg;
  /// }
  /// \endcode
  ///
  /// \param[in] \c arg The argument to update.
  /// \param[in] \c idx The index of the current parameter in the target
  /// function parameter pack, starting from 0. 
  /// \param[in] \c currIdx The index value of the parameter to be updated.
  /// \param[in] \c multiplier A multiplier to compound with h before updating
  /// 'arg'.
  /// \param[out] \c h_val The h value that was set for each argument.
  /// \param[in] \c n The length of the input pointer/array. NOOP here.
  /// \param[in] \c i The specific index to get the adjusted h for. NOOP here.
  ///
  /// \returns The updated argument.
  template <typename T, typename std::enable_if<!std::is_pointer<T>::value,
                                                bool>::type = true>
  T updateIndexParamValue(T arg, std::size_t idx, std::size_t currIdx,
                          int multiplier, precision& h_val, std::size_t n = 0,
                          std::size_t i = 0) {
    if (idx == currIdx) {
      h_val = (h_val == 0) ? get_h(arg) : h_val;
      return arg + h_val * multiplier;
    }
    return arg;
  }

  /// A function to updated the array/pointer arguments given a multiplier and
  /// the argument value. Custom data types can be supported as input in the
  /// case that this function is overloaded.
  ///
  /// This function should be overloaded as follows:
  /// \code
  /// template <typename <--your-data-type-->>
  /// <--your-data-type-->* updateIndexParamValue(<--your-data-type--> arg,
  ///                           std::size_t idx, std::size_t currIdx,
  ///                          int multiplier, precision& h_val,
  ///                          std::size_t n = 0, std::size_t i = 0) {
  ///   if (idx == currIdx) {
  ///     // Add your type specific code here that would essentially translate
  ///     // to `return arg + multiplier * h` (h is some very small value)
  ///     h_val = //Save the 'precision' equivalent value of h here only if it
  ///     is 0, not doing so will lead to undefined behaviour.
  ///   }
  ///   return arg;
  /// }
  /// \endcode
  ///
  /// \warning User function should not delete the returned 'arg' replacement,
  /// it will lead to double free or undefined behaviour.
  ///
  /// \param[in] \c arg The argument to update.
  /// \param[in] \c idx The index of the current parameter in the target
  /// function parameter pack, starting from 0.
  /// \param[in] \c currIdx The index value of the parameter to be
  /// updated.
  /// \param[in] \c multiplier A multiplier to compound with h before
  /// updating 'arg'.
  /// \param[out] \c h_val The h value that was set for each
  /// argument.
  /// \param[in] \c n The length of the input pointer/array.
  /// \param[in] \c i The specific index to get the adjusted h for.
  ///
  /// \returns The updated argument.
  template <typename T>
  T* updateIndexParamValue(T* arg, std::size_t idx, std::size_t currIdx,
                           int multiplier, precision& h_val, std::size_t n = 0,
                           std::size_t i = 0) {
    // malloc some buffer space that we will free later.
    // this is required to make sure that we are retuning a deep copy
    // that is valid throughout the scope of the central_diff function.
    // Temp is system owned.
    T* temp = bufferManager.make_buffer_space<T>(n);
    // deepcopy
    for (std::size_t j = 0; j < n; j++) {
      temp[j] = arg[j];
    }
    if (idx == currIdx) {
      h_val = (h_val == 0) ? get_h(temp[i]) : h_val;
      // update the specific array value.
      temp[i] += h_val * multiplier;
    }
    return temp;
  }

  /// A helper function to calculate the numerical derivative of a target
  /// function.
  ///
  /// \param[in] \c f The target function to numerically differentiate.
  /// \param[out] \c _grad The gradient array reference to which the gradients
  /// will be written.
  /// \param[in] \c printErrors A flag to decide if we want to print numerical
  /// diff errors estimates.
  /// \param[in] \c idxSeq The index sequence associated with
  /// the input parameter pack.
  /// \param[in] \c args The arguments to the function to differentiate.
  template <typename F, std::size_t... Ints,
            typename RetType = typename clad::return_type<F>::type,
            typename... Args>
  void central_difference_helper(
      F f, clad::tape_impl<clad::array_ref<RetType>>& _grad, bool printErrors,
      clad::IndexSequence<Ints...> idxSeq, Args&&... args) {

    std::size_t argLen = sizeof...(Args);
    // loop over all the args, selecting each arg to get the derivative with
    // respect to.
    for (std::size_t i = 0; i < argLen; i++) {
      precision h;
      std::size_t argVecLen = _grad[i].size();
      for (std::size_t j = 0; j < argVecLen; j++) {
        // Reset h
        h = 0;
        // calculate f[x+h, x-h]
        // f(..., x+h,...)
        precision xaf = f(updateIndexParamValue(std::forward<Args>(args), Ints,
                                                i, /*multiplier=*/1, h,
                                                _grad[Ints].size(), j)...);
        precision xbf = f(updateIndexParamValue(std::forward<Args>(args), Ints,
                                                i, /*multiplier=*/-1, h,
                                                _grad[Ints].size(), j)...);
        precision xf1 = (xaf - xbf) / (h + h);

        // calculate f[x+2h, x-2h]
        precision xaf2 = f(updateIndexParamValue(std::forward<Args>(args), Ints,
                                                 i,
                                                 /*multiplier=*/2, h,
                                                 _grad[Ints].size(), j)...);
        precision xbf2 = f(updateIndexParamValue(std::forward<Args>(args), Ints,
                                                 i,
                                                 /*multiplier=*/-2, h,
                                                 _grad[Ints].size(), j)...);
        precision xf2 = (xaf2 - xbf2) / (2 * h + 2 * h);

        if (printErrors) {
          // calculate f(x+3h) and f(x-3h)
          precision xaf3 = f(updateIndexParamValue(std::forward<Args>(args),
                                                   Ints, i,
                                                   /*multiplier=*/3, h,
                                                   _grad[Ints].size(), j)...);
          precision xbf3 = f(updateIndexParamValue(std::forward<Args>(args),
                                                   Ints, i,
                                                   /*multiplier=*/-3, h,
                                                   _grad[Ints].size(), j)...);
          // Error in derivative due to the five-point stencil formula
          // E(f'(x)) = f`````(x) * h^4 / 30 + O(h^5) (Taylor Approx) and
          // f`````(x) = (f[x+3h, x-3h] - 4f[x+2h, x-2h] + 5f[x+h, x-h])/(2 *
          // h^5) Formula courtesy of 'Abramowitz, Milton; Stegun, Irene A.
          // (1970), Handbook of Mathematical Functions with Formulas, Graphs,
          // and Mathematical Tables, Dover. Ninth printing. Table 25.2.`.
          precision error = ((xaf3 - xbf3) - 4 * (xaf2 - xbf2) +
                             5 * (xaf - xbf)) /
                            (60 * h);
          // This is the error in evaluation of all the function values.
          precision evalError = std::numeric_limits<precision>::epsilon() *
                                (std::fabs(xaf2) + std::fabs(xbf2) +
                                 8 * (std::fabs(xaf) + std::fabs(xbf))) /
                                (12 * h);
          // Finally print the error to standard ouput.
          printError(std::fabs(error), evalError, i, argVecLen > 1 ? j : -1);
        }

        // five-point stencil formula = (4f[x+h, x-h] - f[x+2h, x-2h])/3
        _grad[i][j] = 4.0 * xf1 / 3.0 - xf2 / 3.0;
        bufferManager.free_buffer();
      }
    }
  }

  /// A function to calculate the derivative of a function using the central
  /// difference formula. Note: we do not propogate errors resulting in the
  /// following function, it is likely the errors are large enough to be of
  /// significance, hence it is only wise to use these methods when it is
  /// absolutely necessary.
  ///
  /// \param[in] \c f The target function to numerically differentiate.
  /// \param[out] \c _grad The gradient array reference to which the gradients
  /// will be written.
  /// \param[in] \c printErrors A flag to decide if we want to print numerical
  /// diff errors estimates.
  /// \param[in] \c args The arguments to the function to differentiate.
  template <typename F, std::size_t... Ints,
            typename RetType = typename clad::return_type<F>::type,
            typename... Args>
  void central_difference(F f, clad::tape_impl<clad::array_ref<RetType>>& _grad,
                          bool printErrors, Args&&... args) {
    return central_difference_helper(f, _grad, printErrors,
                                     clad::MakeIndexSequence<sizeof...(Args)>{},
                                     std::forward<Args>(args)...);
  }

  /// A helper function to calculate ther derivative with respect to a
  /// single input.
  ///
  /// \param[in] \c f The target function to numerically differentiate.
  /// \param[in] \c arg The argument with respect to which differentiation is
  /// requested.
  /// \param[in] \c n The positional value of 'arg'.
  /// \param[in] \c arrIdx The index value of the input pointer/array to
  /// differentiate with respect to.
  /// \param[in] \c arrLen The length of the pointer/array.
  /// \param[in] \c printErrors A flag to decide if we want to print numerical
  /// diff errors estimates.
  /// \param[in] \c idxSeq The index sequence associated with the input
  /// parameter pack.
  /// \param[in] \c args The arguments to the function to differentiate.
  ///
  /// \returns The numerical derivative
  template <typename F, typename T, std::size_t... Ints, typename... Args>
  precision forward_central_difference_helper(
      F f, T arg, std::size_t n, int arrIdx, std::size_t arrLen,
      bool printErrors, clad::IndexSequence<Ints...> idxSeq, Args&&... args) {

    precision xaf, xbf, xaf2, xbf2, xf1, xf2, dx, h = 0;
    // calculate f[x+h, x-h]
    xaf = f(updateIndexParamValue(std::forward<Args>(args), Ints, n,
                                  /*multiplier=*/1, h, arrLen, arrIdx)...);
    xbf = f(updateIndexParamValue(std::forward<Args>(args), Ints, n,
                                  /*multiplier=*/-1, h, arrLen, arrIdx)...);
    xf1 = (xaf - xbf) / (h + h);

    // calculate f[x+2h, x-2h]
    xaf2 = f(updateIndexParamValue(std::forward<Args>(args), Ints, n,
                                   /*multiplier=*/2, h, arrLen, arrIdx)...);
    xbf2 = f(updateIndexParamValue(std::forward<Args>(args), Ints, n,
                                   /*multiplier=*/-2, h, arrLen, arrIdx)...);
    xf2 = (xaf2 - xbf2) / (2 * h + 2 * h);

    if (printErrors) {
      // calculate f(x+3h) and f(x-3h)
      precision xaf3 = f(
          updateIndexParamValue(std::forward<Args>(args), Ints, n,
                                /*multiplier=*/3, h, arrLen, arrIdx)...);
      precision xbf3 = f(
          updateIndexParamValue(std::forward<Args>(args), Ints, n,
                                /*multiplier=*/-3, h, arrLen, arrIdx)...);
      // Error in derivative due to the five-point stencil formula
      // E(f'(x)) = f`````(x) * h^4 / 30 + O(h^5) (Taylor Approx) and
      // f`````(x) = (f[x+3h, x-3h] - 4f[x+2h, x-2h] + 5f[x+h, x-h])/(2 * h^5)
      // Formula courtesy of 'Abramowitz, Milton; Stegun, Irene A. (1970),
      // Handbook of Mathematical Functions with Formulas, Graphs, and
      // Mathematical Tables, Dover. Ninth printing. Table 25.2.`.
      precision error = ((xaf3 - xbf3) - 4 * (xaf2 - xbf2) + 5 * (xaf - xbf)) /
                        (60 * h);
      // This is the error in evaluation of all the function values.
      precision evalError = std::numeric_limits<precision>::epsilon() *
                            (std::fabs(xaf2) + std::fabs(xbf2) +
                             8 * (std::fabs(xaf) + std::fabs(xbf))) /
                            (12 * h);
      // Finally print the error to standard ouput.
      printError(std::fabs(error), evalError, n, arrIdx);
    }

    // five-point stencil formula = (4f[x+h, x-h] - f[x+2h, x-2h])/3
    dx = 4.0 * xf1 / 3.0 - xf2 / 3.0;
    return dx;
  }

  /// A function to calculate the derivative of a function using the central
  /// difference formula. Note: we do not propogate errors resulting in the
  /// following function, it is likely the errors are large enough to be of
  /// significance, hence it is only wise to use these methods when it is
  /// absolutely necessary. This specific version calculates the numerical
  /// derivative of the target function with respect to only one scalar input
  /// parameter.
  /// \note This function will evaluate to incorrect results if one of the
  /// following conditions are satisfied:
  /// 1) A reference/pointer, which is not 'arg', is being modified in the
  /// function,
  /// 2) A reference/pointer, which is not 'arg', has a length > arrLen.
  /// If this is the case, you may specify the greatest length any non-scalar
  /// input may have.
  /// To overcome these, you may use the central_difference method instead.
  ///
  /// \param[in] \c f The target function to numerically differentiate.
  /// \param[in] \c arg The argument with respect to which differentiation is
  /// requested.
  /// \param[in] \c n The positional value of 'arg'.
  /// \param[in] \c printErrors A flag to decide if we want to print numerical
  /// diff errors estimates.
  /// \param[in] \c args The arguments to the function to differentiate.
  ///
  /// \returns The derivative value.
  template <
      typename F, typename T, typename... Args,
      typename std::enable_if<!std::is_pointer<T>::value, bool>::type = true>
  precision forward_central_difference(F f, T arg, std::size_t n,
                                       bool printErrors, Args&&... args) {
    return forward_central_difference_helper(f, arg, n, /*arrIdx=*/-1,
                                             /*arrLen=*/0, printErrors,
                                             clad::MakeIndexSequence<sizeof...(
                                                 Args)>{},
                                             std::forward<Args>(args)...);
  }

  /// A function to calculate the derivative of a function using the central
  /// difference formula. Note: we do not propogate errors resulting in the
  /// following function, it is likely the errors are large enough to be of
  /// significance, hence it is only wise to use these methods when it is
  /// absolutely necessary. This specific version calculates the numerical
  /// derivative of the target function with respect to only one non-scalar
  /// input parameter.
  /// \note This function will evaluate to incorrect results if one of the
  /// following conditions are satisfied:
  /// 1) A reference/pointer, which is not 'arg', is being modified in the
  /// function,
  /// 2) A reference/pointer, which is not 'arg', has a length > arrLen.
  /// If this is the case, you may specify the greatest length any non-scalar
  /// input may have.
  /// To overcome these, you may use the central_difference method instead.
  ///
  /// \param[in] \c f The target function to numerically differentiate.
  /// \param[in] \c arg The argument with respect to which differentiation is
  /// requested.
  /// \param[in] \c n The positional value of 'arg'.
  /// \param[in] \c arrLen The length of the pointer/array.
  /// \param[in] \c arrIdx The specific index value to differentiate the target
  /// function with respect to.
  /// \param[in] \c printErrors A flag to decide if we want to print numerical
  /// diff errors estimates.
  /// \param[in] \c args The arguments to the function to differentiate.
  ///
  /// \returns The derivative value.
  template <typename F, typename T, typename... Args>
  precision forward_central_difference(F f, T arg, std::size_t n,
                                       std::size_t arrLen, int arrIdx,
                                       bool printErrors, Args&&... args) {
    return forward_central_difference_helper(f, arg, n, arrIdx, arrLen,
                                             printErrors,
                                             clad::MakeIndexSequence<sizeof...(
                                                 Args)>{},
                                             std::forward<Args>(args)...);
  }
} // namespace numerical_diff

#endif // CLAD_NUMERICAL_DIFF_H
#endif // CLAD_NO_NUM_DIFF
