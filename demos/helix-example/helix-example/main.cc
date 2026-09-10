#include <iostream>
#include <cmath>
#include "clad/Differentiator/Differentiator.h"

#include "fitter.h"

int main()
{
	int nr_of_points = 200;
	double points[nr_of_points * 3];
	double a = 5.2122, b = 2, c = 10.835, d = 17.07055, alph = -3.60384, bet = 1.13255;
	GenerateFlawedPoints(nr_of_points, a, b, c, d, alph, bet, points);
	LevenbergMarquardt(points, nr_of_points, b, a, b, c, d, alph, bet);
	// GradientDescent(points, nr_of_points);
	for (int i = 0; i < nr_of_points; i++)
	{
		std::cout << points[i * 3 + 0] << " " << points[i * 3 + 1] << " " << points[i * 3 + 2] << "\n";
	}
	std::cout << "end\n";
	std::cerr << "Results: " << a << " " << b << " " << c << " " << d << " " << alph << " " << bet << std::endl;
}
