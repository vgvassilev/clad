| Linux & OSX: | Coverity |
|:---:|:---:|
[![Linux & Osx Status](http://img.shields.io/travis/vgvassilev/clad.svg?style=flat-square)](https://travis-ci.org/vgvassilev/clad) | <a href="https://scan.coverity.com/projects/vgvassilev-clad"> <img alt="Coverity Scan Build Status" src="https://scan.coverity.com/projects/16418/badge.svg"/> </a>|

##  What is clad
clad is a C++ plugin for clang that implements automatic differentiation of
user-defined functions by employing the chain rule in forward mode, coupled with
source code transformation and AST constant fold.

##  Description
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

##  Building from source LLVM, Clang and clad (development environment)
  ```
    LAST_KNOWN_GOOD_LLVM=$(wget https://raw.githubusercontent.com/vgvassilev/clad/master/LastKnownGoodLLVMRevision.txt -O - -q --no-check-certificate)
    LAST_KNOWN_GOOD_CLANG=$(wget https://raw.githubusercontent.com/vgvassilev/clad/master/LastKnownGoodClangRevision.txt -O - -q --no-check-certificate)
    git clone https://github.com/llvm-mirror/llvm.git src
    cd src; git checkout $LAST_KNOWN_GOOD_LLVM
    cd tools
    git clone https://github.com/llvm-mirror/clang.git clang
    cd clang ; git checkout $LAST_KNOWN_GOOD_CLANG
    cd ../
    git clone https://github.com/vgvassilev/clad.git clad
    cd ../
    cat patches tools/clad/patches/*.diff | patch -p0
    cd ../
    mkdir obj inst
    cd obj
    cmake -DCMAKE_BUILD_TYPE=Debug -DLLVM_TARGETS_TO_BUILD=host -DCMAKE_INSTALL_PREFIX=../inst ../src/
    make && make install
  ```

##  Building from source (example was tested on Ubuntu 18.04 LTS, tests are disabled)
  ```
    # clad is supporting only LLVM/Clang 5.0.x
    sudo apt install clang-5.0 llvm-5.0 clang-5.0-dev llvm-5.0-dev libllvm5.0 llvm-5.0-runtime
    sudo -H pip install lit 
    git clone https://github.com/vgvassilev/clad.git clad
    mkdir build_dir inst; cd build_dir
    cmake ../clad -DLLVM_DIR=/usr/lib/llvm-5.0/lib/cmake/llvm/ -DCMAKE_INSTALL_PREFIX=../inst
    make && make install
  ```

##  Usage
After a successful build libclad.so or libclad.dylib will be created
in llvm's lib (inst/lib) directory. One can attach the plugin to clang invocation
like this:
  ```
 clang -cc1 -x c++ -std=c++11 -load libclad.dylib -plugin clad -plugin-arg-clad -help SourceFile.cpp
  ```
For more details see:  
[ACAT 2014 Slides](https://indico.cern.ch/event/258092/session/8/contribution/90/material/slides/0.pdf)  
[Martin's GSoC2014 Final Report](https://indico.cern.ch/event/337174/contribution/2/material/slides/0.pdf)  
[LLVM Poster](http://llvm.org/devmtg/2013-11/slides/Vassilev-Poster.pdf)  
[Violeta's GSoC2013 Final Report](http://prezi.com/g1iggppw76wl/autodiff/)
[DIANA-HEP Meeting 2018](https://indico.cern.ch/event/760152/contributions/3153263/attachments/1730121/2795841/Clad-DIANA-HEP.pdf)

##  Founders
Founder of the project is Vassil Vassilev as part of his research interests and vision. He holds the exclusive copyright and other related rights, described in Copyright.txt.

##  Contributors
We have quite a few contributors, whose contribution is described briefly in
Credits.txt. If you don't find your name among the list of contributors, please contact us!

##  License
clad is an open source project, licensed by GNU LESSER GENERAL PUBLIC
LICENSE (see License.txt). If there is module with different that LGPL license
it will be explicitly stated in the License.txt in the module's source code
folder.

Please see License.txt for further information.

##  How to Contribute
As stated in 2. clad is an open source project. Like most of the open
source projects we constantly lack of manpower and contributions of any sort are
very welcome. The best starting point is to download the source code and start
playing with it. There are a lot of tests showing implicitly the available
functionality.
