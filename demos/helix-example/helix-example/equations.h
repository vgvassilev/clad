#pragma once

#include <cmath>
#include <cassert>

auto MY_PI = 3.14159265359;

double EvaluateSinPlusLin(double A, double B, double C, double D, double x)
{
	/*When this equation is equal to zero, the distance between the point and the helix is the shortest.*/
	return A * std::sin(x + B) + C * x + D;
}

double SolveSinPlusLin(double A, double B, double C, double D, double mi, double ma)
{
	/*Binary search to determine x, with which EvaluateSinPlusLin equation equals zero.*/
	for (int i = 0; i < 100; i++)
	{
		double mid = (mi + ma) / 2;
		double vmi = EvaluateSinPlusLin(A, B, C, D, mi);
		double vmid = EvaluateSinPlusLin(A, B, C, D, mid);
		double vma = EvaluateSinPlusLin(A, B, C, D, ma);

		if (vmi < 0 and 0 < vmid)
		{
			ma = mid;
		}
		else if (vmid < 0 and 0 < vma)
		{
			mi = mid;
		}
		else if (vmid < 0 and 0 < vmi)
		{
			ma = mid;
		}
		else if (vma < 0 and 0 < vmid)
		{
			mi = mid;
		}
		else
		{
			break;
			mi = mid;
		}
	}

	double x = (mi + ma) / 2;
	return x;
}

double NextValPiK(double offs, double x)
{
	/*Find the next 2 * PI * k + offset (where k is an integer) that is greater than x.*/

	if (x < 0)
	{
		double v = -NextValPiK(-offs, -x) + 2 * MY_PI;
		return v > x ? v : v + 2 * MY_PI;
	}

	double kie = std::floor(x / 2 / MY_PI);

	for (int i = -2; i <= 2; i++)
	{
		double v = (kie + i) * 2 * MY_PI + offs;

		if (v > x)
		{
			return v;
		}
	}

	return 1000000000;
}

// A cos(x + B) + C = 0
double NextSinPlusInflection(double A, double B, double C, double x)
{
	/* Identifies the next inflection point of the sine curve.*/
	// cos(x + B) = -C / A
	if (-C / A >= -1 && -C / A <= 1)
	{
		double inv = std::acos(-C / A);
		return std::min(NextValPiK(inv - B, x), NextValPiK(-inv - B, x));
	}
	else
	{
		return 1000000000;
	}
}
