#include <iostream>
#include <cassert>
#include "matrices.h"
#include "fitter.h"

/* Tests for "matrices.h" */
bool AreSame(double a, double b)
{
    return fabs(a - b) < 0.00009;
}
bool ArraysEqual(double *a, double *b, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (AreSame(a[i], b[i]))
        {
        }
        else
        {
            std::cerr << a[i] << " " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

void TestMultiply()
{
    double a[2 * 3] = {1, 2, 3, 4, 5, 6};
    double b[3 * 4] = {11, 12, 13, 14, 14, 15, 16, 7, 17, 18, 19, 20};
    double output[2 * 4];
    MatrixMultiply(a, b, 2, 3, 4, output);
    double expected_output[2 * 4] = {90, 96, 102, 88, 216, 231, 246, 211};
    for (int i = 0; i < 8; i++)
    {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    assert(ArraysEqual(output, expected_output, 8));
}

void TestTranspose()
{
    double input[2 * 3] = {1, 2, 3, 4, 5, 6};
    double output[3 * 2];
    double expected_output[3 * 2] = {1, 4, 2, 5, 3, 6};
    Transpose(input, 2, 3, output);
    for (int i = 0; i < 6; i++)
    {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    assert(ArraysEqual(output, expected_output, 6));
}

void TestDiagOfSquareM()
{
    double input[3 * 3] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double diag[3 * 3];
    double expected_diag[3 * 3] = {1, 0, 0, 0, 5, 0, 0, 0, 9};
    DiagOfSquareM(input, 3, diag);
    for (int i = 0; i < 9; i++)
    {
        std::cout << diag[i] << " ";
    }
    std::cout << std::endl;
    assert(ArraysEqual(diag, expected_diag, 9));
}

void TestScalarMultiply()
{
    double input[2 * 3] = {1, 2, 3, 4, 5, 6};
    double output[2 * 3];
    double expected_output[2 * 3] = {2, 4, 6, 8, 10, 12};

    ScalarMultiply(input, 2, 3, 2, output);
    for (int i = 0; i < 6; i++)
    {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    assert(ArraysEqual(output, expected_output, 6));
}

void TestAddMatrices()
{
    double a[2 * 3] = {1, 2, 3, 4, 5, 6};
    double b[2 * 3] = {2, 4, 6, 8, 10, 12};
    double output[2 * 3];
    double expected_output[2 * 3] = {3, 6, 9, 12, 15, 18};
    AddMatrices(a, b, 2, 3, output);
    for (int i = 0; i < 6; i++)
    {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    assert(ArraysEqual(output, expected_output, 6));
}

void TestGaussianElim()
{
    double left_side[36] = {

        2, 3, 1, 5, 7, 1, //

        4,
        7, 2, 10, 14, 2, //

        1,
        2, 2, 3, 5, 2, //

        3,
        5, 4, 1, 6, 4, //

        5,
        1, 3, 2, 1, 3, //

        2,
        4, 6, 1, 3, 5};

    double right_side[6] = {25, 53, 18, 31, 23, 40};
    double output[36];
    double results[6];
    ForwardElim(left_side, 6, right_side, output);
    std::cout << "right side " << std::endl;
    for (int i = 0; i < 6; i++)
    {
        std::cout << right_side[i] << " ";
    }
    std::cout << std::endl;

    BackSub(output, 6, right_side, results);
    for (int i = 0; i < 6; i++)
    {
        std::cout << results[i] << " ";
    }
    std::cout << std::endl;
}

void TestDistanceToPoint()
{
    double a = 5.2122, b = -4.79395, c = -26.40835, d = -4.207055, alph = -3.60384, bet = 1.13255;
    double t = 0.1;
    double output[3];
    HelixPoint(a, b, c, d, alph, bet, t, output);
    std::cout << "Generated point " << output[0] << " " << output[1] << " " << output[2] << " " << std ::endl;
    double x = output[0], y = output[1], z = output[2];
    double dist = DistanceToPoint(a, b, c, d, alph, bet, x, y, z);
    std::cout << "Distance to point on helix is: " << dist << std ::endl;
}

void TestDistancesToPoints()
{
    double a = 5.2122, b = -4.79395, c = -26.40835, d = -4.207055, alph = -3.60384, bet = 1.13255;
    double t = 0;
    double points[10][3];

    std::cout << "Generated points: " << std::endl;
    for (int i = 0; i < 10; i++)
    {
        t += 0.1;
        double output[3];
        HelixPoint(a, b, c, d, alph, bet, t, output);
        double x = output[0], y = output[1], z = output[2];
        points[i][0] = x;
        points[i][1] = y;
        points[i][2] = z;
        std::cout << output[0] << " " << output[1] << " " << output[2] << " " << std ::endl;
    }
    double points1D[10 * 3];
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            points1D[i * 3 + j] = points[i][j];
        }
    }
    double dist[10 * 3];
    DistancesToAllPoints(points1D, 10, a, b, c, d, alph, bet, dist);
    std::cout << "Distances to all points: " << std::endl;
    for (int i = 0; i < 10; i++)
    {
        std::cout << dist[i] << std::endl;
    }
    std::cout << "done\n";
}

int main()
{
    std::cout << "TestMultiply results: " << std::endl;
    TestMultiply();
    std::cout << "TestTranspose results: " << std::endl;
    TestTranspose();
    std::cout << "TestDiagOfSquareM results: " << std::endl;
    TestDiagOfSquareM();
    std::cout << "TestScalarMultiply results: " << std::endl;
    TestScalarMultiply();
    std::cout << "TestAddMatrices results: " << std::endl;
    TestAddMatrices();
    std::cout << "TestGaussianElim results: " << std::endl;
    TestGaussianElim();
    std::cout << "TestDistanceToPoint results: " << std::endl;
    TestDistanceToPoint();
    std::cout << "TestDistancesToAllPoint results: " << std::endl;
    TestDistancesToPoints();
}