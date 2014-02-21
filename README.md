0. What is clad
clad is a C++ plugin for clang that implements automatic differentiation of 
user-defined functions by employing the chain rule in forward mode, coupled with
source code transformation and AST constant fold.

1. Description
In mathematics and computer algebra, automatic differentiation (AD) is a set of 
techniques to numerically evaluate the derivative of a function specified by a 
computer program. Automatic differentiation is an alternative technique to 
Symbolic differentiation and Numerical differentiation (the method of finite 
differences) that yields exact derivatives even of complicated functions.
The goal of the presented plugin is to extend the Cling functionality in order 
to make it possible for the tool to differentiate non-trivial functions and 
find partial derivatives for trivial cases. Our implementation approach is to 
employ source code transformation, which consists of explicitly building a 
new source code through a compiler-like process that includes parsing the 
original program, constructing an internal representation, and performing 
global analysis. This elegant but laborious process is greatly aided by 
Cling (http://cern.ch/cling) which does not only provide the necessary facilities
 for code transformation, but also serves as a basis for the plugin.

2. Building from source
  svn checkout http://llvm.org/svn/llvm-project/llvm/trunk src
  cd src/tools
  svn checkout http://llvm.org/svn/llvm-project/cfe/trunk clang
  git clone https://github.com/vgvassilev/clad.git clad
  cd ../
  cat patches tools/clad/patches/*.diff | patch -p0
  cd ../
  mkdir obj inst
  cd obj
  ../src/configure --prefix=../inst
  make && make install

3. Usage
  After a successful build libAutoDiff.so or libAutoDiff.dylib will be created
in llvm's lib (inst/lib) directory. One can attach the plugin to clang invocation
like this:

 clang -cc1 -x c++ -std=c++11 -load libAutoDiff.dylib -plugin clad -plugin-arg-clad -fprint-folded-fn -plugin-arg-clad -fprint-folded-fn-ast SourceFile.cpp

For more details see: http://llvm.org/devmtg/2013-11/slides/Vassilev-Poster.pdf
