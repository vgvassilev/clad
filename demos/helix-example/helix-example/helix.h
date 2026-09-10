#pragma once

#include <cmath>
#include "rotations.h"
#include "equations.h"
#include "distance.h"

inline void HelixPoint(double a, double b, double c, double d, double alph, double bet, double t, double output[3])
{
	/*Describe a point on a helix in the Cartesian coordinate system.*/
	double x = a * (c + std::cos(t));
	double y = a * (d + std::sin(t));
	double z = a * b * t;
	output[0] = x;
	output[1] = y;
	output[2] = z;
	Rotate(x, y, z, alph, bet, output);
}

inline double HelixClosestTime(double a, double b, double c, double d, double alph, double bet, double x, double y, double z)
{
	/*Calculate t, during which a helix with given params is the closest to a given point.*/
	auto MY_PI = 3.14159265359;
	double point[3];
	UnRotate(x, y, z, alph, bet, point);
	point[0] /= a;
	point[1] /= a;
	point[2] /= a;
	point[0] -= c;
	point[1] -= d;
	double A = std::sqrt(point[0] * point[0] + point[1] * point[1]);
	double B = std::atan2(-point[1], point[0]);
	double C = b * b;
	double D = -point[2] * b;

	double mi = point[2] / b - MY_PI;
	double ma = point[2] / b + MY_PI;
	double t1 = SolveSinPlusLin(A, B, C, D, mi, ma);

	double ans = t1;
	HelixPoint(a, b, c, d, alph, bet, ans, point);
	double dist = DistanceSquareA(point, x, y, z);

	for (double t = mi; t < ma; t = t)
	{
		double ttt = NextSinPlusInflection(A, B, C, t);

		if (ttt == t)
		{
			break;
		}

		double cur = SolveSinPlusLin(A, B, C, D, t, ttt);
		t = ttt;
		HelixPoint(a, b, c, d, alph, bet, cur, point);
		double dist2 = DistanceSquareA(point, x, y, z);

		if (dist2 < dist)
		{
			dist = dist2;
			ans = cur;
		}
	}

	return ans;
}
