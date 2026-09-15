#pragma once

#include <cmath>

inline void RotateAlph(double x, double y, double z, double alph, double output[3]) {
	output[0] = x;
	output[1] = y * cos(alph) - z * sin(alph);
	output[2] = y * sin(alph) + z * cos(alph);
}

inline void RotateBet(double x, double y, double z, double bet, double output[3]) {
	output[0] = x * cos(bet) + z * sin(bet);
	output[1] = y;
	output[2] = -x * sin(bet) + z * cos(bet);
}

inline void Rotate(double x, double y, double z, double alph, double bet, double output[3]) {
	double point[3];
	RotateAlph(x, y, z, alph, point);
	RotateBet(point[0], point[1], point[2], bet, output);
}

inline void UnRotate(double x, double y, double z, double alph, double bet, double output[3]) {
	double point[3];
	RotateBet(x, y, z, -bet, point);
	RotateAlph(point[0], point[1], point[2], -alph, output);
}
