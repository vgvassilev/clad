# Using your custom estimation model with clad 
Clad's error estimation framework provides users with a choice to write their own custom error models and use those to generate estimation code. By default clad uses the Taylor series approximation model to estimate errors in a function, however this model might not be the most suitable alternative in many cases. In such scenarios, a user may prefer to write their own model to get better fp error estimates. 

The aim of this demo is to illustrate how one can integrate a custom model with clad. For in depth information on how to *write* your own custom estimation model, check out [this tutorial](https://compiler-research.org/tutorials/fp_error_estimation_clad_tutorial/).

## Building the demo

Before we can use the custom model, it must be compiled into a [shared object](https://www.thegeekstuff.com/2012/06/linux-shared-libraries/). To do this, you can use your favorite compiler. For this demo, we will be using the clang compiler.

Firstly, we shall set up some environment variables to simplify following the rest of the tutorial.

After building the code as specified in the [README.md](https://github.com/vgvassilev/clad#how-to-install), run the following command to set environment variables which we will use later:

```bash
$ export CLAD_INST=$PWD/../inst;
$ export CLAD_BASE=$PWD/../clad;
```

> **TIP**: You can put the above lines in your ~/.bashrc or equivalent shell "rc" file to maintain the same variables across multiple sessions.

Now, in a terminal, run the following:

```bash
$ clang++ -ICLAD_INST/include -fPIC -shared -fno-rtti -Wl,-undefined -Wl,suppress CLAD_BASE/demos/CustomModel/CustomModel.cpp -o libCustomModel.so
``` 
 The above should create a `libCustomModel.so` in the same directory you executed that command in. Once the shared object is created, we are ready to run it with clad.

## Running the demo

Now, to use your custom estimation model, you can just specify the `.so` created in the previous section as an input to clad via CLI. The specific parameters you would need to add are given below:

```bash
-Xclang -plugin-arg-clad -Xclang -fcustom-estimation-model -Xclang -plugin-arg-clad -Xclang ./libCustomModel.so
``` 
So a typical invocation to clad would then look like the following:

```bash
clang++ -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang CLAD_INST/lib/clad.so -ICLAD_INST/include -Xclang -plugin-arg-clad -Xclang -fcustom-estimation-model -Xclang -plugin-arg-clad -Xclang ./libCustomModel.so CLAD_BASE/demos/CustomModel/test.cpp
```
## Verifying results

To verify your results, you can build the dummy `test.cpp` file with the commands shown above. Once you compile and run the test file correctly, you will notice the generated code is as follows:

```cpp
The code is: void func_grad(float x, float y, float *_d_x, float *_d_y, double &_final_error) {
    float _d_z = 0;
    float _t0;
    float z;
    _t0 = z;
    z = x + y;
    _d_z += 1;
    {
        _final_error += _d_z * z;
        z = _t0;
        float _r_d0 = _d_z;
        _d_z = 0;
        *_d_x += _r_d0;
        *_d_y += _r_d0;
    }
    _final_error += *_d_x * x;
    _final_error += *_d_y * y;
 }
```

Here, notice that the result in the `_final_error` variable  now reflects the error expression defined in the custom model we just compiled!

This demo is also a runnable test under `CLAD_BASE/test/Misc/RunDemos.C` and will run as a part of the lit test suite. Thus, the same can be verified by running `make check-clad`.
