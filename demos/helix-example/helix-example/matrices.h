#pragma once
#include <cassert>
#include <cmath>
#include <iostream>

// All the matrices are written as 1D arrays!
inline void MatrixMultiply(double *a, double *b, int arows, int acols, int bcols, double *output)
{
    for (int i = 0; i < arows; i++)
    {
        for (int j = 0; j < bcols; j++)
        {
            double sum = 0;
            for (int k = 0; k < acols; k++)
                sum = sum + a[i * acols + k] * b[k * bcols + j];
            output[i * bcols + j] = sum;
        }
    }
}

inline void Transpose(double *input, int rows, int cols, double *output)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            int i_input = i * cols + j;

            int i_output = j * rows + i;

            output[i_output] = input[i_input];
        }
    }
}

inline void DiagOfSquareM(double *input, int height, double *diag)
{
    // Works for square matrices only
    for (int i = 0; i < height * height; i++)
    {
        diag[i] = 0;
    }
    for (int i = 0; i < height; i++)
    {
        diag[i * height + i] = input[i * height + i];
    }
}

inline void ScalarMultiply(double *matrix, int rows, int cols, double number, double *output)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            output[i * cols + j] = number * matrix[i * cols + j];
        }
    }
}

inline double VectorLen(double *vector, int size)
{
    double length = 0;
    for (int i = 0; i < size; i++)
    {
        length += vector[i] * vector[i];
    }
    return std::sqrt(length);
}
inline void CopyMatrix(double *matrix, int size, double *output)
{
    for (int i = 0; i < size; i++)
    {
        output[i] = matrix[i];
    }
}
inline void AddMatrices(double *a, double *b, int rows, int cols, double *output)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            output[i * cols + j] = a[i * cols + j] + b[i * cols + j];
        }
    }
}

inline void swap_row(double *matrix, int size, int i, int j)
{
    for (int k = 0; k < size; k++)
    {
        double temp = matrix[i * size + k];
        matrix[i * size + k] = matrix[j * size + k];
        matrix[j * size + k] = temp;
    }
}
inline void ForwardElim(double *input, int size, double *res, double *output)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            output[i * size + j] = input[i * size + j];
        }
    }
    for (int i = 0; i < size; i++)
    {
        int i_max = i;
        double v_max = output[i_max * size + i];

        for (int j = i + 1; j < size; j++)
            if (std::abs(output[j * size + i]) > std::abs(v_max) && output[j * size + i] != 0)
                v_max = output[j * size + i], i_max = j;
        if (i_max != i)
        {
            swap_row(output, size, i, i_max);
            double temp = res[i];
            res[i] = res[i_max];
            res[i_max] = temp;
        }

        if (output[i * size + i] == 0.0)
        {
            std::cerr << "Mathematical Error!";
            std::cerr << "Input that caused the error is:";
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    std::cerr << output[i * size + j] << " ";
                }
                std::cerr << std::endl;
            }
        }
        for (int j = i + 1; j < size; j++)
        {
            double ratio = output[j * size + i] / output[i * size + i];

            for (int k = 0; k < size; k++)
            {
                output[j * size + k] = output[j * size + k] - ratio * output[i * size + k];
                if (std::abs(output[j * size + k]) <= 1e-15)
                {
                    output[j * size + k] = 0;
                }
            }
            res[j] = res[j] - ratio * res[i];
            if (std::abs(res[j]) <= 1e-15)
            {
                res[j] = 0;
            }
        }
    }

    // std::cerr << "Forward elimination results:" << std::endl;
    // std::cerr << "Left side:" << std::endl;
    // for (int j = 0; j < size; j++)
    // {
    //     for (int k = 0; k < size; k++)
    //     {
    //         std::cerr << output[j * size + k] << " ";
    //     }
    //     std::cerr << std::endl;
    // }
    // std::cerr << "Right side:" << std::endl;
    // for (int k = 0; k < size; k++)
    // {
    //     std::cerr << res[k] << " ";
    // }
    // std::cerr << std::endl;
}

void BackSub(double *input, int size, double *right_side, double *results)
{
    /*Back substitution and the result of Gaussian elimination*/
    for (int i = (size - 1); i > -1; i--)
    {
        results[i] = right_side[i];
        for (int j = (size - 1); j > i; j--)
        {
            results[i] -= input[i * size + j] * results[j];
        }
        results[i] /= input[i * size + i];
        if (std::abs(results[i]) <= 1e-15)
        {
            results[i] = 0;
        }
    }
}
void CheckSolution(double *input, int size, double *right_side, double *results)
{
    for (int i = 0; i < size; i++)
    {
        double sum = 0;
        for (int j = 0; j < size; j++)
        {
            sum += input[i * size + j] * results[j];
        }
        if (std::abs(sum - right_side[i]) >= 1e-5)
        {
            std::cerr << "Wrong solution " << sum << " " << right_side[i] << std::endl;
            std::abort();
        }
    }
}

inline void PrintMatrix(std::string name, double *matrix, int rows, int cols)
{
    std::cerr << name << std::endl;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            std::cerr << matrix[i * cols + j] << " ";
        }
        std::cerr << std::endl;
    }
}