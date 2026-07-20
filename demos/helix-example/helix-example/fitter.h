#include <iostream>
#include <random>
#include <ctime>
#include <cassert>
#include <fstream>

#include "helix.h"
#include "equations.h"
#include "matrices.h"
#include "clad/Differentiator/Differentiator.h"

double DistanceToPoint(double a, double b, double c, double d, double alph, double bet, double x, double y, double z)
{
    /*Calculate the distance to a single point. */
    double t = HelixClosestTime(a, b, c, d, alph, bet, x, y, z);
    double output[3];
    HelixPoint(a, b, c, d, alph, bet, t, output);
    double dist = DistanceA(output, x, y, z);
    dist += 0.001 * ((a * a) + (b * b) + (c * c) + (d * d) + (alph * alph) + (bet * bet));

    return dist;
}

double SquareErr(double *points, int nr_of_points, double a, double b, double c, double d, double alph, double bet)
{
    /*Calculate the residual sum of squares. */
    double dist;
    double square_err = 0;
    for (int i = 0; i < nr_of_points; i++)
    {
        double x = points[i * 3];
        double y = points[i * 3 + 1];
        double z = points[i * 3 + 2];
        dist = DistanceToPoint(a, b, c, d, alph, bet, x, y, z);
        square_err += (dist * dist);
    }
    return square_err;
}

void Points(int nr_of_points, double a, double b, double c, double d, double alph, double bet)
{
    /*Generate and print out points on a helix with given params. */
    double t = 0;
    for (int i = 0; i < nr_of_points; i++)
    {
        t += 0.1;
        double output[3];
        HelixPoint(a, b, c, d, alph, bet, t, output);
        double x = output[0], y = output[1], z = output[2];
        std::cout << x << " " << y << " " << z << "\n";
    }
    std::cout << "end\n";
}

void GenerateFlawedPoints(int nr_of_points, double a, double b, double c, double d, double alph, double bet, double *points)
{
    /*Generate points on a helix with given params but add noise. */
    auto seed = time(nullptr);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(-2 * MY_PI, 2 * MY_PI);
    double output[3];
    double t = 0;
    for (int i = 0; i < nr_of_points; i++)
    {
        t += 0.1;
        HelixPoint(a, b, c, d, alph, bet, t, output);
        points[i * 3] = output[0] + uniform(rng) / 10;
        points[i * 3 + 1] = output[1] + uniform(rng) / 10;
        points[i * 3 + 2] = output[2] + uniform(rng) / 10;
    }
}

void DistancesToAllPoints(double *points, int nr_of_points, double a, double b, double c, double d, double alph, double bet, double *dist)
{
    /*Calculate the distances to all points. */
    int n = 0;
    for (int i = 0; i < nr_of_points; i++)
    {
        double x = points[i * 3];
        double y = points[i * 3 + 1];
        double z = points[i * 3 + 2];
        dist[n] = DistanceToPoint(a, b, c, d, alph, bet, x, y, z);
        n++;
    }
}
void Jacobian(double *points, int nr_of_points, double a, double b, double c, double d, double alph, double bet, double *Jacobian)
{
    /*Construct the nr_of_points x 6 Jacobian.*/
    auto dist_grad = clad::gradient(DistanceToPoint, "a, b, c, d, alph, bet");
    for (int i = 0; i < nr_of_points; i++)
    {
        double x = points[i * 3];
        double y = points[i * 3 + 1];
        double z = points[i * 3 + 2];
        double output[3];
        double da = 0, db = 0, dc = 0, dd = 0, dalph = 0, dbet = 0;
        dist_grad.execute(a, b, c, d, alph, bet, x, y, z, &da, &db, &dc, &dd, &dalph, &dbet);
        Jacobian[i * 6] = da;
        Jacobian[i * 6 + 1] = db;
        Jacobian[i * 6 + 2] = dc;
        Jacobian[i * 6 + 3] = dd;
        Jacobian[i * 6 + 4] = dalph;
        Jacobian[i * 6 + 5] = dbet;
    }
}

double Lambda(double *points, int nr_of_points, double &a, double &b, double &c, double &d, double &alph, double &bet, double lambda, double &square_err, double *results)
{
    /*Calculate the damping coefficient lambda for the next iteration of the LevenbergMarquardt function.*/
    double new_lambda;
    double new_square_err = SquareErr(points, nr_of_points, a + results[0], b + results[1], c + results[2], d + results[3], alph + results[4], bet + results[5]);
    // std::cerr << "SQUARE ERR " << new_square_err << std::endl;
    if ((new_square_err >= square_err) && (lambda < 1000))
        new_lambda = lambda * 10;
    else
    {
        // std::cerr << "IMPROVEMENTS!";
        a += results[0];
        b += results[1];
        c += results[2];
        d += results[3];
        alph += results[4];
        bet += results[5];
        new_lambda = lambda / 10;
        square_err = new_square_err;
    }
    return new_lambda;
}

void LevenbergMarquardt(double *points, int nr_of_points, double true_b, double &a, double &b, double &c, double &d, double &alph, double &bet)
/*Use the Levenberg-Marquardt algorithm to fit a helix on a given set of points. Currently produces all of the parameters of the helix, except b.*/
{
    a = 6.2122, b = 0.1, c = 1.9835, d = 1.707055, alph = -3.60384, bet = 1.13255; // currently breaks if the parameters are exact as the ones used for (noise free) generated points

    int diff_params = 6;
    double lambda = 1;
    double lambda_change = 1;
    double square_err;
    double jacobian[nr_of_points * diff_params];
    double tjacobian[diff_params * nr_of_points];
    double tjj[diff_params * diff_params];
    double results[diff_params];
    double counter = 0;
    {
        double dist[nr_of_points];
        DistancesToAllPoints(points, nr_of_points, a, b, c, d, alph, bet, dist);
        square_err = 0;
        for (int i = 0; i < nr_of_points; i++)
        {
            square_err += (dist[i] * dist[i]);
        }
    }

    for (int i = 0; i < 200; i++)
    {

        Jacobian(points, nr_of_points, a, b, c, d, alph, bet, jacobian);

        Transpose(jacobian, nr_of_points, diff_params, tjacobian);

        MatrixMultiply(tjacobian, jacobian, diff_params, nr_of_points, diff_params, tjj);

        double diag[diff_params * diff_params];
        DiagOfSquareM(tjj, diff_params, diag);

        double identity[diff_params * diff_params];
        ScalarMultiply(diag, diff_params, diff_params, lambda, identity);
        double left_side[diff_params * diff_params];
        AddMatrices(tjj, identity, diff_params, diff_params, left_side);
        double dist[nr_of_points];
        DistancesToAllPoints(points, nr_of_points, a, b, c, d, alph, bet, dist);
        double right_side[diff_params * 1];
        MatrixMultiply(tjacobian, dist, diff_params, nr_of_points, 1, right_side);
        ScalarMultiply(right_side, 1, diff_params, -1, right_side);

        // left side is 6x6, right side is 6x1, so h is 6x1.
        double forward_elim[diff_params * diff_params];
        double unchanged_rs[diff_params];
        CopyMatrix(right_side, diff_params, unchanged_rs);
        ForwardElim(left_side, diff_params, right_side, forward_elim);
        BackSub(forward_elim, diff_params, right_side, results);
        CheckSolution(left_side, diff_params, unchanged_rs, results);
        double old_square_err = square_err;
        lambda = Lambda(points, nr_of_points, a, b, c, d, alph, bet, lambda, square_err, results);
        if (int(square_err) == int(old_square_err) && counter > 10 && square_err < old_square_err)
            break;
        else if (int(square_err) == int(old_square_err))
            counter++;
        else
            counter = 0;
        old_square_err = square_err;

        // std::cerr << "New params: " << a << " " << b << " " << c << " " << d << " " << alph << " " << bet << " ";
        // std::cerr << "lambda: " << lambda << " squares distance: " << square_err << std::endl;
    }
    b = true_b;
    Points(nr_of_points, a, b, c, d, alph, bet);
}

void GradientDescent(double *points, int nr_of_points)
{
    /*Implementation of the gradient descent algorithm. Gets stuck in a local minimum.*/
    double a = 5.2122, b = 0.1, c = 0.9835, d = 1.707055, alph = -3.60384, bet = 1.13255;
    double lambda = 0.00001;
    double jacobian[nr_of_points * 6];
    double tjacobian[6 * nr_of_points];
    double dist[nr_of_points];
    double square_err = SquareErr(points, nr_of_points, a, b, c, d, alph, bet);
    double params[6] = {0};
    double prev_square_er = SquareErr(points, nr_of_points, a, b, c, d, alph, bet);
    std::cerr << square_err << std::endl;
    for (int i = 0; i < 2000; i++)
    {
        DistancesToAllPoints(points, nr_of_points, a, b, c, d, alph, bet, dist);
        Jacobian(points, nr_of_points, a, b, c, d, alph, bet, jacobian);
        Transpose(jacobian, nr_of_points, 6, tjacobian);

        double y_dist[nr_of_points];
        ScalarMultiply(dist, nr_of_points, 1, -1, y_dist);
        double h[6];
        MatrixMultiply(tjacobian, y_dist, 6, nr_of_points, 1, h);
        ScalarMultiply(h, 6, 1, lambda, h);
        double new_square_err = SquareErr(points, nr_of_points, a + h[0], b + h[1], c + h[2], d + h[3], alph + h[4], bet + h[5]);

        if (new_square_err < prev_square_er)
        {
            lambda = lambda * 10;
        }
        else
        {
            lambda = lambda / 10;
            continue;
        }
        a += h[0];
        b += h[1];
        c += h[2];
        d += h[3];
        alph += h[4];
        bet += h[5];
        if (new_square_err < square_err)
        {
            square_err = new_square_err;
            params[0] = a;
            params[1] = b;
            params[2] = c;
            params[3] = d;
            params[4] = alph;
            params[5] = bet;
        }
        prev_square_er = new_square_err;
        // std::cerr << "New params: " << a << " " << b << " " << c << " " << d << " " << alph << " " << bet << " ";
        // std::cerr << "lambda: " << lambda << " squares distance: " << new_square_err << std::endl;
    }

    double t = -nr_of_points / 2;
    for (int i = 0; i < 10 * nr_of_points; i++)
    {
        t += 0.1;
        double output[3];
        HelixPoint(params[0], params[1], params[2], params[3], params[4], params[5], t, output);
        double x = output[0], y = output[1], z = output[2];

        std::cout << x << " " << y << " " << z << "\n";
    }
    std::cout << "end\n";
}