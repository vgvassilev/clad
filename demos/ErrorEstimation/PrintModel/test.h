#include <cstdlib>
#include <iostream>
#include <limits>

namespace clad
{
    __attribute__((always_inline)) double getErrorVal(double dx, double x, const char *name)
    {
        double error = std::abs(dx * x * std::numeric_limits<float>::epsilon());
        std::cout << "Error in " << name << " : " << error << std::endl;
        return error;
    }
} // clad
