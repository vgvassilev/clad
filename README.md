<div align="center">

<img width=200em alt="Clad" src="https://user-images.githubusercontent.com/6516307/193281076-3f5a4c7f-2f0d-4c05-b76b-b2d6719adf0a.svg"/>

[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/clad)](https://github.com/conda-forge/clad-feedstock)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/clad/badges/license.svg)](https://anaconda.org/conda-forge/clad)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/clad/badges/platforms.svg)](https://anaconda.org/conda-forge/clad)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/clad/badges/downloads.svg)](https://anaconda.org/conda-forge/clad)

[![Linux & Osx Status](https://github.com/vgvassilev/clad/workflows/Main/badge.svg)](https://github.com/vgvassilev/clad/actions?query=workflow%3AMain) <a href="https://scan.coverity.com/projects/vgvassilev-clad"> <img alt="Coverity Scan Build Status" src="https://scan.coverity.com/projects/16418/badge.svg"/> </a>
[![codecov]( https://codecov.io/gh/vgvassilev/clad/branch/master/graph/badge.svg)](https://codecov.io/gh/vgvassilev/clad)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vgvassilev/clad/master?labpath=%2Fdemos%2FJupyter%2FIntro.ipynb)


Clad is a source-transformation [automatic differentiation (AD)](https://en.wikipedia.org/wiki/Automatic_differentiation) library for C++,<br/>implemented as a plugin for the [Clang compiler](http://clang.llvm.org/). 

#### [Try Online](https://godbolt.org/z/3KWhY4j8M) | [Binder](https://mybinder.org/v2/gh/vgvassilev/clad/master?labpath=%2Fdemos%2FJupyter%2FIntro.ipynb) | [Usage](#how-to-use-clad) | [Installation](#how-to-install) | [Further Reading](#further-reading) | [Documentation](https://clad.readthedocs.io/en/latest/index.html) | [Contributing](#how-to-contribute)
</div>

## About Clad
Clad enables automatic differentiation (AD) for C++. It is based on LLVM compiler infrastructure and is a plugin for Clang compiler. Clad is based on source code transformation. Given C++ source code of a mathematical function, it can automatically generate C++ code for computing derivatives of the function. It supports both forward-mode and reverse-mode AD.Clad has extensive coverage of modern C++ features and a robust fallback and recovery system in place.

## How to use Clad

Clad provides five API functions:
- [`clad::differentiate`](#forward-mode---claddifferentiate) to use forward-mode AD.
- [`clad::gradient`](#reverse-mode---cladgradient) to use reverse-mode AD.
- [`clad::hessian`](#hessian-mode---cladhessian) to compute Hessian matrix using a combination of forward-mode and reverse-mode AD.
- [`clad::jacobian`](#jacobian-mode---cladjacobian) to compute Jacobian matrix using reverse-mode AD.
- [`clad::estimate-error`](#floating-point-error-estimation---cladestimate_error) to compute the floating-point error of the given program using reverse-mode AD.

API functions are used to label an existing function for differentiation.
Both functions return a functor object containing the generated derivative which can be called via `.execute` method, which forwards provided arguments to the generated derivative function.

For a guide on compiling your clad-based programs, look [here](#compiling-and-executing-your-code-with-clad).

### Forward mode - `clad::differentiate`
For a function `f` of several inputs and single (scalar) output, forward mode AD can be used to compute (or, in case of Clad, create a function) a directional derivative of `f` with respect to a *single* specified input variable. Derivative function created by the forward-mode AD is guaranteed to have *at most* a constant factor (around 2-3) more arithmetical operations compared to the original function.

`clad::differentiate(f, ARGS)` takes 2 arguments:
1. `f` is a pointer to a function or a method to be differentiated
2. `ARGS` is either:
  * a single numerical literal indicating an index of independent variable (e.g. `0` for `x`, `1` for `y`)
  * a string literal with the name of independent variable (as stated in the *definition* of `f`, e.g. `"x"` or `"y"`), and if the variable is an array the index needs to be specified, e.g. `"arr[1]"`

Generated derivative function has the same signature as the original function `f`, however its return value is the value of the derivative. Example:
```cpp
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

double f(double x, double y) { return x * y; }

int main() {
  // Call clad to generate the derivative of f wrt x.
  auto f_dx = clad::differentiate(f, "x");
  // Execute the generated derivative function.
  std::cout << f_dx.execute(/*x=*/3, /*y=*/4) << std::endl;
  // Dump the generated derivative code to standard output.
  f_dx.dump();
}
```

### Reverse mode - `clad::gradient`
Reverse-mode AD allows computing the gradient of `f` using *at most* a constant factor (around 4) more arithmetical operations compared to the original function. While its constant factor and memory overhead is higher than that of the forward-mode, it is independent of the number of inputs. E.g. for a function having N inputs and consisting of T arithmetical operations, computing its gradient takes a single execution of the reverse-mode AD and around 4\*T operations, while it would take N executions of the forward-mode, this requiring up to N\*3\*T operations.

`clad::gradient(f, /*optional*/ ARGS)` takes 1 or 2 arguments:
1. `f` is a pointer to a function or a method to be differentiated
2. `ARGS` is either:
  * not provided, then `f` is differentiated w.r.t. its every argument
  * a string literal with comma-separated names/indices of independent variables (e.g. `"x"`, `"y"`, `"x, y"`, `"y, x"`, "0, 1", "0, y", etc.)
  * a SINGLE number representing the index of the independent variable

Since a vector of derivatives must be returned from a function generated by the reverse mode, its signature is slightly different. The generated function has `void` return type and same input arguments. The function has additional `n` arguments (where `n` refers to the number of arguments whose gradient was requested) of type `T*`, where `T` is the type of the corresponding original variable. Each of these variables stores the derivative of the elements as they appear in the orignal function signature. *The caller is responsible for allocating and zeroing-out the gradient storage*. Example:
```cpp
auto f_grad = clad::gradient(f);
double dx = 0, dy = 0;
// After this call, dx and dy will store the derivatives of x and y respectively.
f_grad.execute(x, y, &dx, &dy);
std::cout << "dx: " << dx << ' ' << "dy: " << dy << std::endl;

// Same effect as before.
auto f_dx_dy = clad::gradient(f, "x, y"); 
auto f_dy_dx = clad::gradient(f, "y, x");

// The same effect can be achieved by using an array instead of individual variables.
double result2[2] = {};
f_dy_dx.execute(x, y, /*dx=*/&result2[0], /*dy=*/&result2[1]);
// note that the derivatives are mapped to the "result" indices in the same order as they were specified in the argument:
std::cout << "dy: " << result2[0] << ' ' << "dx: " << result2[1] << std::endl;
```
### Hessian mode - `clad::hessian`

Clad can produce the hessian matrix of a function using its forward and reverse mode capabilities. 
Its interface is similar to reverse mode but differs when arrays are involved. It returns the matrix as a flattened 
vector in row major format.

`clad::hessian(f, /*optional*/ ARGS)` takes 1 or 2 arguments:
1. `f` is a pointer to a function or a method to be differentiated
2. `ARGS` is either:
    * not provided, then `f` is differentiated w.r.t. its every argument except in the case of arrays where it needs to
      be provided
    * a string literal with comma-separated names of independent variables (e.g. `"x"` or `"y"` or `"x, y"` or `"y, x"`
      or in case of arrays `"x[0:2]"`) 

The generated function has `void` return type and same input arguments. The function has an additional argument of 
type `T*`, where `T` is the return type of `f`. This variable stores the hessian
matrix. *The caller is responsible for allocating and zeroing-out the hessian storage*. Example:
```cpp
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

double f(double x, double y) { return x * y; }
double g(double x, double y[2]) { return x * y[0] * y[1]; }

int main() {
    // Since we are differentiating variables that are not arrays the interface
    // is same as in reverse mode
    auto f_hess = clad::hessian(f);
    // The size of the resultant matrix should be the square of the 
    // number of independent variables
    double mat_f[4] = {0};
    
    // Execute the hessian function
    f_hess.execute(/*x=*/3, /*y=*/4, mat_f);
    std::cout << "[" << mat_f[0] << ", " << mat_f[1] << "\n  " 
                     << mat_f[2] << ", " << mat_f[3] << "]";
    
    // When arrays are involved the array indexes that are to be differentiated needs to be specified
    // even if the whole array is being differentiated
    auto g_hess = clad::hessian(g, "x, y[0:1]");
    // The rest of the steps are the same.
}
```

### Jacobian mode - `clad::jacobian`

Clad can produce the jacobian of a function using its *vectorized forward mode*. It returns the jacobian matrix as a `clad::matrix` for every pointer/array parameter.

`clad::jacobian(f, /*optional*/ ARGS)` takes 1 or 2 arguments:
1. `f` is a pointer to a function or a method to be differentiated
2. `ARGS` is either:
    * not provided, then `f` is differentiated w.r.t. its every argument
    * a string literal with comma-separated names of independent variables (e.g. `"x"` or `"y"` or `"x, y"` or `"y, x"`)

The generated function has `void` return type and same input arguments. For every pointer/array parameter `arr`, the function has an additional argument `_d_vector_arr`. Its
type is `clad::matrix<T>`, where `T` is the pointee type of `arr`. These variables store their derivatives w.r.t. all inputs. Output parameters are supposed to have `_clad_out_` prefix.
*The caller is responsible for allocating the matrices*. Example:

```cpp
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

void h(double a, double b, double _clad_out_output[]) {
    output[0] = a * a * a;
    output[1] = a * a * a + b * b * b;
    output[2] = 2 * (a + b);
}

int main() {
    // This sets all the input variables (i.e a, b, and output) as independent variables 
    auto h_jac = clad::jacobian(h);
    
    // The jacobian matrix size should be
    // the size of the output x the number of independent variables
    // In this case it is 3 x (1 + 1 + 3)
    clad::matrix<double> d_output(3, 5);
    double output[3] = {0};
    h_jac.execute(/*a=*/3, /*b=*/4, output, &d_output);

    // d_output[i][j] is the derivative of the i-th element of `output` w.r.t. the j-th input
    std::cout << d_output[0][0] << " " << d_output[0][1] << std::endl
              << d_output[1][0] << " " << d_output[1][1] << std::endl
              << d_output[2][0] << " " << d_output[2][1] << std::endl;
}
```

Or in the case of multiple array parameters:

```cpp
#include "clad/Differentiator/Differentiator.h"
#include <iostream>

void h(double a, double b, double _clad_out_arr[], double* _clad_out_ptr) {
    arr[0] = a * a * a;
    ptr[0] = arr[0] + b * b * b;
    arr[1] = 2 * (a + b);
}

int main() {
    auto h_jac = clad::jacobian(h);

    // The jacobian matrix size should be
    // the size of the output x the number of independent variables

    // 3 x (1 + 1 + 2 + 1)
    clad::matrix<double> d_arr(2, 5);
    double arr[2] = {0};

    // 1 x (1 + 1 + 2 + 1)
    clad::matrix<double> d_ptr(1, 5);
    double ptr[1] = {0};

    h_jac.execute(/*a=*/3, /*b=*/4, arr, ptr, &d_arr, &d_ptr);
    
    // d_arr[i][j] is the derivative of the i-th element of `arr` w.r.t. the j-th input
    std::cout << d_arr[0][0] << " " << d_arr[0][1] << std::endl
              << d_arr[1][0] << " " << d_arr[1][1] << std::endl;

    // Likewise, with `ptr`
    std::cout << d_ptr[0][0] << " " << d_ptr[0][1] << std::endl;
}
```

### Floating-point error estimation - `clad::estimate_error`

Clad is capable of annotating a given function with floating point error estimation code using the reverse mode of AD. An interface similar to `clad::gradient(f)` is provided as follows:

`clad::estimate_error(f)` takes 1 argument:
1. `f` is a pointer to the function or method to be annotated with floating point error estimation code.

The function signature of the generated code is the same as from `clad::gradient(f)` with the exception that it has an extra argument at the end of type `double&`, which returns the total floating point error in the function by reference. For a user function `double f(double, double)` example usage is described below:

```cpp
// Generate the floating point error estimation code for 'f'.
auto df = clad::estimate_error(f);
// Print the generated code to standard output.
df.dump();
// Declare the necessary variables.
double x, y, d_x, d_y, final_error = 0;
// Finally call execute on the generated code.
df.execute(x, y, &d_x, &d_y, final_error);
// After this, 'final_error' contains the floating point error in function 'f'.
```
The above example generates the the error code using an in-built taylor approximation model. However, clad is capable of using any user defined custom model, for information on how to use you own custom model, please visit [this demo](https://github.com/vgvassilev/clad/tree/master/demos/ErrorEstimation/CustomModel).

More detail on the APIs can be found under clad's [user documentation](https://clad.readthedocs.io/en/latest/index.html).
### Compiling and executing your code with clad

#### Using Jupyter Notebooks

[xeus-cpp](https://github.com/compiler-research/xeus-cpp) provides a Jupyter kernel for C++ with the help of the C++ interpreter clang-repl and the native implementation of the Jupyter protocol xeus. Within the xeus-cpp framework, Clad can enable automatic differentiation (AD) such that users can automatically generate C++ code for their computation of derivatives of their functions.

To set up your environment, use:

```
mamba create -n xeus-clad -c conda-forge clad xeus-cpp clangdev=20 jupyterlab

conda activate xeus-clad
```

Next, running `jupyter notebook` will show 3 new kernels for `C++ 11/14/17` with Clad attached.

Try out a Clad [tutorial](https://compiler-research.org/tutorials/clad_jupyter/) interactively in your browser through binder: 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vgvassilev/clad/master?labpath=%2Fdemos%2FJupyter%2FIntro.ipynb)

#### Using as a plugin for Clang
Since Clad is a Clang plugin, it must be properly attached when Clang compiler is invoked. First, the plugin must be built to get `libclad.so` (or `.dylib`).

To compile `SourceFile.cpp` with Clad enabled, use the following commands:

- Clang++: `clang++ -std=c++11 -I /full/path/to/include/ -fplugin=/full/path/to/lib/clad.so Sourcefile.cpp`
- Clang: `clang -x c++ -std=c++11 -I /full/path/to/include/ -fplugin=/full/path/to/lib/clad.so SourceFile.cpp -lstdc++ -lm`

Clad also provides certain flags to save and print the generated derivative code:

- To save the Clad generated derivative code to `Derivatives.cpp`: `-Xclang -plugin-arg-clad -Xclang -fgenerate-source-file`
- To print the Clad generated derivative: `-Xclang -plugin-arg-clad -Xclang -fdump-derived-fn`

## How to install
At the moment, LLVM/Clang 8.0.x - 18.1.x are supported.

### Conda Installation

Clad is available using [conda](https://anaconda.org/conda-forge/clad):

```bash
conda install -c conda-forge clad
```

If you have already added `conda-forge` as a channel, the `-c conda-forge` is unnecessary. Adding the channel is recommended because it ensures that all of your packages use compatible versions:

```bash
conda config --add channels conda-forge
conda update --all
```

### Building from source (example was tested on Ubuntu 20.04 LTS)
```
#sudo apt install clang-11 libclang-11-dev llvm-11-tools llvm-11-dev
sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
sudo -H pip install lit
git clone https://github.com/vgvassilev/clad.git clad
mkdir build_dir inst; cd build_dir
cmake ../clad -DClang_DIR=/usr/lib/llvm-11 -DLLVM_DIR=/usr/lib/llvm-11 -DCMAKE_INSTALL_PREFIX=../inst -DLLVM_EXTERNAL_LIT="$(which lit)"
make && make install
```

> **NOTE**: On some Linux distributions (e.g. Arch Linux), the LLVM and Clang libraries are installed at `/usr/lib/cmake/llvm` and `/usr/lib/cmake/clang`. If compilation fails with the above provided command, ensure that you are using the correct path to the libraries.

###  Building from source (example was tested on macOS Big Sur 11.6)
```
brew install llvm@12
brew install python
python -m pip install lit
git clone https://github.com/vgvassilev/clad.git clad
mkdir build_dir inst; cd build_dir
cmake ../clad -DLLVM_DIR=/opt/homebrew/opt/llvm@12/lib/cmake/llvm -DClang_DIR=/opt/homebrew/opt/llvm@12/lib/cmake/clang -DCMAKE_INSTALL_PREFIX=../inst  -DLLVM_EXTERNAL_LIT="`which lit`"
make && make install
make check-clad
```

> **NOTE**: If you are using clad as a clang plugin after building it from source, please make sure that you uses the same compiler version you built clad against. Apple distributed clang does not work because Apple has disabled clang plugins.

### Developer Environment - Build LLVM, Clang and Clad from source:

```
pip3 install lit
```
Clone the LLVM project and checkout the required LLVM version (Currently supported versions 8.x - 18.x)

```
git clone https://github.com/llvm/llvm-project.git
git clone https://github.com/vgvassilev/clad.git
cd llvm-project
git checkout release/18.x
```

Build Clad with Clang and LLVM:
```
mkdir build && cd build
cmake -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_EXTERNAL_PROJECTS=clad -DLLVM_EXTERNAL_CLAD_SOURCE_DIR=../../clad -DCMAKE_BUILD_TYPE="Debug" -DLLVM_TARGETS_TO_BUILD=host -DLLVM_INSTALL_UTILS=ON ../llvm
cmake --build . --target clad --parallel $(nproc --all)
```
Note: However, on some systems, the above command may not build Clang and Clad in parallel. In such cases, you need to build Clang separately and then build Clad using that Clang binary. Use the following commands:

Building Clang Separately
```
mkdir build && cd build
cmake -DLLVM_ENABLE_PROJECTS="clang" -DCMAKE_BUILD_TYPE=DEBUG -DLLVM_TARGETS_TO_BUILD=host -DLLVM_INSTALL_UTILS=ON ../llvm
cmake --build . --target clang --parallel $(nproc --all)
make -j8 check-clang  # This installs llvm-config, required by lit
cd ../..

```
Cloning and Building Clad:
```
cd clad
mkdir build && cd build
cmake -DLLVM_DIR=PATH/TO/llvm-project/build -DCMAKE_BUILD_TYPE=DEBUG -DLLVM_EXTERNAL_LIT="$(which lit)" ../
make -j8 clad


```
If you have limited memory (less than 16GB of RAM + swap), you can use the following build configuration to compile Clang more efficiently:
```
cmake -G Ninja /path/to/llvm-project/llvm -DLLVM_USE_LINKER=gold -DCMAKE_BUILD_TYPE=Debug -DLLVM_TARGETS_TO_BUILD=host -DBUILD_SHARED_LIBS=On -DLLVM_USE_SPLIT_DWARF=On -DLLVM_OPTIMIZED_TABLEGEN=On -DLLVM_ENABLE_PROJECTS=clang -DCMAKE_INSTALL_PREFIX=../inst
```


Run the Clad tests:
```
make -j8 check-clad
```

## Further reading

### What can be differentiated
Clad is based on compile-time analysis and transformation of C++ abstract syntax tree (Clang AST). This means that Clad must be able to see the body of a function to differentiate it (e.g. if a function is defined in an external library there is no way for Clad to get its AST).



Note: Clad currently differentiates types such as `int`/`char`/`boolean` as any real type (such as `float`, `double`, etc.) would be differentiated. Users should keep in mind that Clad *does not* warn against lossy casts, which on differentiation may result in incorrect derivatives.

Note: If for any reason clad is unable to algorithmically differentiate a function, it automatically switches to numerically differentiating the same. To disable this behavior, please compile your programs with the `-DCLAD_NO_NUM_DIFF` flag. The numerical differentiation functionality can also be used standalone over a wide range of function signatures with minimal user intervention. [This presentation](https://indico.cern.ch/event/1066812/contributions/4495279/attachments/2301763/3915404/Numerical%20Differentiaition%20.pdf) provides more information on what can be numerically differentiated. For a comprehensive demo on using custom user defined types with numerical differentiation, you can check out [this demo](https://github.com/vgvassilev/clad/blob/master/demos/CustomTypeNumDiff.cpp).

### Specifying custom derivatives
Sometimes Clad may be unable to differentiate your function (e.g. if its definition is in a library and source code is not available). Alternatively, an efficient/more numerically stable expression for derivatives may be know. In such cases, it is useful to be able to specify a custom derivatives for your function.

Clad supports that functionality by allowing to specify your own derivatives in `namespace clad::custom_derivatives`. For a function named `FNAME` you can specify:
* a custom derivative w.r.t `I`-th argument by defining a function `FNAME_dargI` inside `namespace clad::custom_derivatives`
* a custom gradient w.r.t every argument by defining a function `FNAME_grad` inside `namespace clad::custom_derivatives`

When Clad will encounter a function `FNAME`, it will first do a lookup inside the `clad::custom_derivatives` namespace to try to find a suitable custom function, and only if none is found will proceed to automatically derive it.

Example:
* Suppose that you have a function `my_pow(x, y)` which computes `x` to the power of `y`. However, Clad is not able to differentiate `my_pow`'s body (e.g. it calls an external library or uses some non-differentiable approximation):
```cpp
double my_pow(double x, double y) { // something non-differentiable here... }
```
However, you know analytical formulas of its derivatives, and you can easily specify custom derivatives:
```cpp
namespace clad::custom_derivatives {
  double my_pow_darg0(double x, double y) { return y * my_pow(x, y - 1); }
  double my_pow_darg1(double x, double y) { return my_pow(x, y) * std::log(x); }
}
```
You can also specify a custom gradient:
```cpp
namespace clad::custom_derivatives {
  void my_pow_grad(double x, double y, double* _d_x, double* _d_y) {
     double t = my_pow(x, y - 1);
     *_d_x = y * t;
     *_d_y = x * t * std::log(x);
   }
}
```
Whenever Clad will encounter `my_pow` inside differentiated function, it will find and use provided custom functions instead of attempting to differentiate it.

Note: Clad provides custom derivatives for some mathematical functions from `<cmath>` inside `clad/Differentiator/BuiltinDerivatives.h`.

Details on custom derivatives, other supported C++ syntax (already supported or in-progress) and further resources can be found over at clad's [user documentation](https://clad.readthedocs.io/en/latest/index.html).

  ## Citing Clad
```latex
% Peer-Reviewed Publication
%
% 16th International workshop on Advanced Computing and Analysis Techniques
% in physics research (ACAT), 1-5 September, 2014, Prague, The Czech Republic
%
@inproceedings{Vassilev_Clad,
  author = {Vassilev,V. and Vassilev,M. and Penev,A. and Moneta,L. and Ilieva,V.},
  title = {{Clad -- Automatic Differentiation Using Clang and LLVM}},
  journal = {Journal of Physics: Conference Series},
  year = 2015,
  month = {may},
  volume = {608},
  number = {1},
  pages = {012055},
  doi = {10.1088/1742-6596/608/1/012055},
  url = {https://iopscience.iop.org/article/10.1088/1742-6596/608/1/012055/pdf},
  publisher = {{IOP} Publishing}
}
```

##  Founders
Founder of the project is Vassil Vassilev as part of his research interests and vision. He holds the exclusive copyright and other related rights, described in Copyright.txt.

##  License
clad is an open source project, licensed by GNU LESSER GENERAL PUBLIC
LICENSE (see License.txt). If there is module with different that LGPL license
it will be explicitly stated in the License.txt in the module's source code
folder.

Please see License.txt for further information.

##  How to Contribute

We are always looking for improvements to the tool, as such open-source
developers are greatly appreciated! If you are interested in getting started
with contributing to clad, make sure you checkout our
[contribution guide](CONTRIBUTING.md).
