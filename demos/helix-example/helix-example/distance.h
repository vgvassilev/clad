#pragma once

#include <cmath>

inline double DistanceSquare(double x1, double y1, double z1, double x2, double y2, double z2)
{
	return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
}

inline double Distance(double x1, double y1, double z1, double x2, double y2, double z2)
{
	return std::sqrt(DistanceSquare(x1, y1, z1, x2, y2, z2));
}

inline double DistanceSquareA(double v[3], double x2, double y2, double z2)
{
	return DistanceSquare(v[0], v[1], v[2], x2, y2, z2);
}

inline double DistanceA(double v[3], double x2, double y2, double z2)
{
	return Distance(v[0], v[1], v[2], x2, y2, z2);
}