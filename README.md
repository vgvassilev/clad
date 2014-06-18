1. What is clad  
clad is a C++ plugin for clang that implements automatic differentiation of 
user-defined functions by employing the chain rule in forward mode, coupled with
source code transformation and AST constant fold.

2. Description  
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

3. Building from source  
  ```
    LAST_KNOWN_GOOD_LLVM=$(wget https://raw.githubusercontent.com/vgvassilev/clad/master/LastKnownGoodLLVMRevision.txt -O - -q)
    svn checkout http://llvm.org/svn/llvm-project/llvm/trunk -r$LAST_KNOWN_GOOD_LLVM src
    cd src/tools
    svn checkout http://llvm.org/svn/llvm-project/cfe/trunk -r$LAST_KNOWN_GOOD_LLVM clang
    git clone https://github.com/vgvassilev/clad.git clad
    cd ../
    cat patches tools/clad/patches/*.diff | patch -p0
    cd ../
    mkdir obj inst
    cd obj
    ../src/configure --prefix=../inst
    make && make install
  ```

4. Usage  
After a successful build libclad.so or libclad.dylib will be created
in llvm's lib (inst/lib) directory. One can attach the plugin to clang invocation
like this:

 clang -cc1 -x c++ -std=c++11 -load libclad.dylib -plugin clad -plugin-arg-clad -help SourceFile.cpp  
For more details see:  
http://llvm.org/devmtg/2013-11/slides/Vassilev-Poster.pdf  
http://prezi.com/g1iggppw76wl/autodiff/  

2. Founders  
Founder of the project is Vassil Vassilev as part of his research interests and vision. He holds the exclusive copyright and other related rights, described in Copyright.txt.

3. Contributors  
We have quite a few contributors, whose contribution is described briefly in
Credits.txt. If you don't find your name among the list of contributors, please
contact us!

4. License  
clad is an open source project, licensed by GNU LESSER GENERAL PUBLIC 
LICENSE (see License.txt). If there is module with different that LGPL license
it will be explicitly stated in the License.txt in the module's source code 
folder. 
  Please see License.txt for further information.

5. How to Contribute  
As stated in 2. clad is an open source project. Like most of the open 
source projects we constantly lack of manpower and contributions of any sort are
very welcome. The best starting point is to download the source code and start
playing with it. There are a lot of tests showing implicitly the available 
functionality.
