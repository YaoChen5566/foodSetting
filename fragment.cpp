#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>

#include "fragment.h"

frag::frag(int _r, int _q, int _l)
{
	setInfo(_r, _q, _l);
}

void frag::setInfo(int rS, int qS, int mL)
{
	r = rS;
	q = qS;
	l = mL;
}