# Using your custom estimation model with clad 
Clad's error estimation framework provides users with a choice to write their own custom error models and use those to generate estimation code. By default clad uses the Taylor series approximation model to estimate errors in a function, however this model might not be the most sutiable alternative in many cases. In such scenarios, a user may prefer to write their own model to get better fp error estimates. 

The aim of this demo is to illustrate how one can integrate a custom model with clad. For in depth information on how to *write* your own custom estimation model, check out the tutorials here.
 <!--- TODO: Add doc link -->

## Building the demo

Before we can use the custom model, it must be compiled into a [shared object](https://www.thegeekstuff.com/2012/06/linux-shared-libraries/). To do this, you can use your favourite compiler. For this demo, we will be using the clang compiler.

In a terminal, run the following:

```bash
$ /path/to/clang -Ipath/to/install/include -fPIC -shared -fno-rtti -Wl,-undefined -Wl,suppress path/to/CustomModel.cpp -o libCustomModel.so
``` 
 The above should create a ```libCustomModel.so``` in the same directory you executed that command in. Once the shared object is created, we are ready to run it with clad.

## Running the demo

Now, to use your custom estimation model, you can just specify the ```.so``` created in the previous section as an input to clad via CLI. The specific parameters you would need to add are given below:

```bash
-Xclang -plugin-arg-clad -Xclang -fcustom-estimation-model -Xclang -plugin-arg-clad -Xclang ./libCustomModel.so
``` 
So a typical invocation to clad would then look like the following:

```bash
./bin/clang -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang path/to/clad.so -I../include/ -x c++ -lstdc++ -Xclang -plugin-arg-clad -Xclang -fcustom-estimation-model -Xclang -plugin-arg-clad -Xclang ./libCustomModel.so /path/to/some.cpp
```
## Verifying results

To verify your results, you can build the dummy ```test.cpp``` file with the commands shown above. Once you compile and run the test file correctly, you will notice the generated code is as follows:

```cpp
The code is: void func_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
    double _delta_z = 0;
    float _EERepl_z0;
    float _d_z = 0;
    float _EERepl_z1;
    float z;
    _EERepl_z0 = z;
    z = x + y;
    _EERepl_z1 = z;
    float func_return = z;
    goto _label0;
  _label0:
    _d_z += 1;
    {
        float _r_d0 = _d_z;
        * _d_x += _r_d0;
        * _d_y += _r_d0;
        _delta_z += _r_d0 * _EERepl_z1;
        _d_z -= _r_d0;
    }
    double _delta_x = 0;
    _delta_x += * _d_x * x;
    double _delta_y = 0;
    _delta_y += * _d_y * y;
    _final_error += _delta_y + _delta_z + _delta_x;
}
```

Here, notice that the result in the ```_delta_z``` variable  now reflects the error expression defined in the custom model we just compiled!

> For information on how to use clad functions and more dev related information, check out our docs here! <<!--TODO: Add doc link>>
